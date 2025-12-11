"""Texo - LaTeX OCR GUI Application."""
import atexit
import base64
import logging
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

# Environment Configuration
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

import gradio as gr
import torch
import latex2mathml.converter
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel

# --- Path Setup ---
PROJECT_ROOT = Path(__file__).parent
TEXO_ROOT = PROJECT_ROOT / "Texo"
SRC_PATH = TEXO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# --- Imports from src ---
try:
    import texo.utils.config  # noqa: F401
    from texo.data.processor import EvalMERImageProcessor
except ImportError as e:
    print(f"Error importing texo modules: {e}")
    sys.exit(1)

# --- Configuration & Constants ---
DEFAULT_MODEL_DIR = TEXO_ROOT / "model"
ASSETS_DIR = TEXO_ROOT / "assets"
LOGO_FILE = ASSETS_DIR / "svg" / "logo.svg"
IMAGE_SIZE = {"width": 384, "height": 384}

EXAMPLE_IMAGES = [
    str(TEXO_ROOT / "TechnoSelection" / "test_img" / "单行公式.png"),
    str(TEXO_ROOT / "TechnoSelection" / "test_img" / "单行公式2.png"),
    str(TEXO_ROOT / "TechnoSelection" / "test_img" / "多行公式.png"),
    str(TEXO_ROOT / "TechnoSelection" / "test_img" / "多行公式2.jpg"),
]

# LaTeX to MathML replacements
LATEX_TO_MATHML_REPLACEMENTS = {
    r"\rarr": r"\to", r"\infin": r"\infty", r"\ge": r"\geq", r"\le": r"\leq",
    r"\gt": ">", r"\lt": "<", r"\cdotp": r"\cdot",
    r"\lvert": "|", r"\rvert": "|", r"\lVert": "||", r"\rVert": "||",
    r"\Bbb": r"\mathbb", r"\bold": r"\mathbf", r"\rm": r"\mathrm", 
    r"\it": r"\mathit", r"\bf": r"\mathbf", r"\overline": r"\bar", "~": r"\quad"
}

# --- UI Resources ---
DROPDOWN_HTML = """
<div style="display: flex; gap: 10px; width: 100%;">
    <div class="custom-dropdown">
        <div class="custom-dropbtn">
            <span style="display: flex; align-items: center; gap: 6px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>
                Copy LaTeX
            </span>
            <span style="font-size: 10px;">▼</span>
        </div>
        <div class="custom-dropdown-content">
            <div onclick="copyLatexToClipboard('raw')">Raw</div>
            <div onclick="copyLatexToClipboard('$')">$ ... $</div>
            <div onclick="copyLatexToClipboard('$$')">$$ ... $$</div>
            <div onclick="copyLatexToClipboard('\\[')">\\[ ... \\]</div>
            <div onclick="copyLatexToClipboard('\\(')">\\( ... \\)</div>
            <div onclick="copyLatexToClipboard('equation')">Equation Env</div>
        </div>
    </div>
    <div class="custom-dropdown">
        <div class="custom-dropbtn">
            <span style="display: flex; align-items: center; gap: 6px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                Copy MathML
            </span>
            <span style="font-size: 10px;">▼</span>
        </div>
        <div class="custom-dropdown-content">
            <div onclick="requestConversion('word')">For Word</div>
            <div onclick="requestConversion('ascii')">AsciiMath</div>
            <div onclick="requestConversion('typst')">Typst</div>
            <div onclick="requestConversion('docx')">Export Docx</div>
        </div>
    </div>
</div>
"""

UI_CSS = """
.header { text-align: center; padding: 1.5rem; }
.header-title { display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
.header h1 { margin: 0; font-size: 2rem; }
.header p { margin: 0.5rem 0 0; color: #666; }
.output-box textarea, #conversion-result textarea { 
    font-family: Consolas, Monaco, monospace !important; 
    max-height: 200px !important; 
    overflow-y: auto !important; 
}
.preview-box { min-height: 150px !important; max-height: 300px !important; overflow-y: auto !important; padding: 1rem !important; border: 1px solid #e0e0e0 !important; border-radius: 8px !important; }
.footer { text-align: center; padding: 1rem; color: #888; font-size: 0.9rem; }
.footer a { color: #2563EB; }

/* Toast Notification */
.toast-container { position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%); z-index: 9999; pointer-events: none; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.toast { background: linear-gradient(135deg, #FF512F, #DD2476); color: #fff; padding: 12px 24px; border-radius: 6px; opacity: 0; transition: opacity 0.3s ease-in-out; box-shadow: 0 4px 12px rgba(0,0,0,0.2); font-size: 14px; text-align: center; font-weight: 500; }
.toast.show { opacity: 1; }

/* Custom Dropdown */
.custom-dropdown { position: relative; display: inline-block; width: 100%; font-family: "Source Sans Pro", ui-sans-serif, system-ui, sans-serif; }
.custom-dropdown::after { content: ""; position: absolute; left: 0; right: 0; bottom: -20px; height: 20px; background: transparent; }
.custom-dropbtn { background-color: white; color: #374151; padding: 10px 12px; font-size: 14px; border: 1px solid #e5e7eb; border-radius: 8px; cursor: pointer; width: 100%; text-align: left; display: flex; justify-content: space-between; align-items: center; transition: border-color 0.2s; }
.custom-dropdown:hover .custom-dropbtn { border-color: #f97316; }
.custom-dropdown-content { display: none; position: absolute; background-color: white; min-width: 100%; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); z-index: 9999; border-radius: 8px; border: 1px solid #e5e7eb; overflow: hidden; margin-top: 4px; }
.custom-dropdown:hover .custom-dropdown-content { display: block; }
.custom-dropdown-content div { color: #374151; padding: 8px 12px; text-decoration: none; display: block; cursor: pointer; font-size: 14px; }
.custom-dropdown-content div:hover { background-color: #f3f4f6; color: #f97316; }
.hidden-box { display: none !important; }
.dropdown-container { overflow: visible !important; position: relative; z-index: 100; }
"""

UI_JS = """
<script>
function showToast(message) {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    container.appendChild(toast);
    
    // Force reflow
    void toast.offsetWidth;
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

function getTextareaValue(id) {
    const el = document.querySelector(`#${id} textarea`);
    return el ? el.value : "";
}

function copyLatexToClipboard(format) {
    const latex = getTextareaValue('latex-output');
    if (!latex) { showToast("Nothing to copy"); return; }
    
    let text = latex;
    const formats = {
        "raw": latex,
        "$": "$" + latex + "$",
        "$$": "$$" + latex + "$$",
        "\\[": "\\\\[" + latex + "\\\\]",
        "\\(": "\\\\(" + latex + "\\\\)",
        "equation": "\\\\begin{equation}\\n" + latex + "\\n\\\\end{equation}"
    };
    if (formats[format]) text = formats[format];
    
    // Optional: Update result box to show what was copied
    const resultBox = document.querySelector('#conversion-result textarea');
    if (resultBox) {
        resultBox.value = text;
        resultBox.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    navigator.clipboard.writeText(text).then(
        () => { showToast("LaTeX Copied"); },
        (err) => { showToast("Copy failed: " + err); }
    );
}

function requestConversion(action) {
    const formatInput = document.querySelector('#hidden-format textarea');
    const convertBtn = document.getElementById('hidden-convert-btn');
    
    if (formatInput && convertBtn) {
        formatInput.value = action;
        formatInput.dispatchEvent(new Event('input', { bubbles: true }));
        convertBtn.click();
        // showToast("Converting...");
    }
}

function handleConversionResult(content, action) {
    if (!content) { showToast("Conversion failed or empty"); return; }
    
    if (action === "word") {
        // Create HTML blob for Word to recognize MathML
        const blob = new Blob([content], {type: 'text/html'});
        const item = new ClipboardItem({'text/html': blob});
        navigator.clipboard.write([item]).then(
            () => { showToast("MathML Copied (Paste in Word)"); },
            (err) => { showToast("Copy failed: " + err); }
        );
    } else if (action === "ascii") {
        navigator.clipboard.writeText(content).then(
            () => { showToast("AsciiMath Copied"); },
            (err) => { showToast("Copy failed: " + err); }
        );
    } else if (action === "typst") {
        navigator.clipboard.writeText(content).then(
            () => { showToast("Typst Code Copied"); },
            (err) => { showToast("Copy failed: " + err); }
        );
    } else if (action === "docx") {
        navigator.clipboard.writeText(content).then(
            () => { showToast("Docx Has Been Exported"); },
            (err) => { showToast("Copy failed: " + err); }
        );
    } else {
        navigator.clipboard.writeText(content).then(
            () => { showToast("Copied " + action); },
            (err) => { showToast("Copy failed: " + err); }
        );
    }
}
</script>
"""

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class _AsyncioFilter(logging.Filter):
    """Filter out specific asyncio errors that are noisy but harmless."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(x in msg for x in ("ConnectionResetError", "_call_connection_lost", "WinError"))

logging.getLogger("asyncio").addFilter(_AsyncioFilter())

# --- Helper Functions ---

@lru_cache(maxsize=1)
def _load_logo_content() -> str:
    """Load the SVG logo content."""
    if not LOGO_FILE.exists():
        return ""
    return LOGO_FILE.read_text(encoding="utf-8")

def get_logo_html(size: int) -> str:
    """Generate HTML for the logo."""
    svg = _load_logo_content()
    if not svg:
        return ""
    # Inject width/height into the svg tag
    svg = svg.replace("<svg ", '<svg width="100%" height="100%" ', 1)
    return f'<span style="display:inline-block;width:{size}px;height:{size}px;vertical-align:middle;">{svg}</span>'

def get_favicon_html() -> str:
    """Generate HTML for the favicon."""
    svg = _load_logo_content()
    if not svg:
        return ""
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{encoded}">'

def _normalize_latex_for_mathml(text: str) -> str:
    """Standardize LaTeX commands for better MathML conversion."""
    # 1. Fix operatorname variants for standard operators
    for op in ["lim", "min", "max", "sup", "inf"]:
        text = re.sub(r"\\operatorname\*?\{" + op + r"\}", lambda m: "\\" + op, text)

    # 2. Common replacements
    for old, new in LATEX_TO_MATHML_REPLACEMENTS.items():
        # Use regex to ensure we match whole commands
        pattern = re.compile(re.escape(old) + r"(?![a-zA-Z])")
        text = pattern.sub(lambda m: new, text)

    # 3. Fix remaining operatorname*
    text = text.replace(r"\operatorname*", r"\operatorname")
    
    return text

def convert_latex_to_mathml(latex: str) -> str:
    """Safely convert LaTeX to MathML."""
    if not latex or latex.startswith("⚠️") or latex.startswith("("):
        return ""
    try:
        latex = _normalize_latex_for_mathml(latex)
        # Use display="block" for better rendering
        return latex2mathml.converter.convert(latex, display="block")
    except Exception as e:
        logger.warning(f"MathML conversion failed: {e}")
        return ""

def format_latex_for_display(latex: str) -> str:
    """Format LaTeX for the preview window."""
    if not latex or latex.startswith("⚠️") or latex.startswith("("):
        return f"*{latex}*" if latex else ""
    return f"$$\n{latex.strip()}\n$$"

# --- Model Wrapper ---

class TexoModelWrapper:
    """Wrapper for the Texo VisionEncoderDecoderModel."""

    def __init__(self) -> None:
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.tokenizer = None
        self.processor: Optional[EvalMERImageProcessor] = None
        self.device: Optional[torch.device] = None
        self._error: Optional[str] = None
        atexit.register(self.unload_model)

    def unload_model(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self) -> bool:
        """Load the model, tokenizer, and processor."""
        if self.model:
            return True
        
        if not DEFAULT_MODEL_DIR.exists():
            self._error = f"Model directory not found: {DEFAULT_MODEL_DIR}"
            logger.error(self._error)
            return False
            
        try:
            start_time = time.time()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading model on device: %s", self.device)
            
            self.model = VisionEncoderDecoderModel.from_pretrained(str(DEFAULT_MODEL_DIR)).to(self.device).eval() # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_MODEL_DIR))
            self.processor = EvalMERImageProcessor(image_size=IMAGE_SIZE)
            
            logger.info("Model loaded in %.2fs", time.time() - start_time)
            return True
        except Exception as e:
            self._error = str(e)
            logger.exception("Failed to load model: %s", e)
            self.unload_model()
            return False

    def _clean_latex_output(self, text: str) -> str:
        """
        Clean up spaces and special characters in the result.
        Heuristic: Protects "\cmd char" spacing, removes all other spaces, then restores protected ones.
        """
        # 1. Protect "\cmd char" (e.g. "\alpha b")
        text = re.sub(r"(\\[a-zA-Z]+)\s+([a-zA-Z])", r"\1_TEXO_SP_\2", text)
        # 2. Remove all spaces
        text = text.replace(" ", "")
        # 3. Restore protected spaces
        return text.replace("_TEXO_SP_", " ")

    def predict(self, image: Optional[Image.Image]) -> str:
        """Run inference on an image."""
        if not self.model or not self.processor:
            return f"⚠️ {self._error or 'Model not loaded'}"
        if not image:
            return ""
            
        try:
            # Preprocessing: Ensure RGB
            if image.mode != "RGB":
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    bg = Image.new("RGB", image.size, (255, 255, 255))
                    bg.paste(image, mask=image.convert("RGBA").split()[-1])
                    image = bg
                else:
                    image = image.convert("RGB")
            
            # Inference
            pixel_values = self.processor(image).unsqueeze(0).to(self.device) # type: ignore
            with torch.no_grad():
                output_ids = self.model.generate(pixel_values=pixel_values)
            
            result = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip() # type: ignore
            return self._clean_latex_output(result) or "(No content detected)"
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "⚠️ Out of memory"
        except Exception as e:
            logger.exception("Prediction error: %s", e)
            return f"⚠️ {e}"

    @property
    def device_status(self) -> str:
        """Return a status string for the UI."""
        if not self.model:
            return f"Error: {self._error[:30]}..." if self._error else "Not loaded"
        
        device_name = "GPU" if self.device.type == "cuda" else "CPU" # type: ignore
        if self.device.type == "cuda": # type: ignore
            device_name += f": {torch.cuda.get_device_name(0)}"
        return f"Ready ({device_name})"

# --- UI Construction ---

def build_ui(model_wrapper: TexoModelWrapper) -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Texo", css=UI_CSS, head=get_favicon_html() + UI_JS) as demo:
        # Header
        gr.HTML(f'<div class="header"><div class="header-title">{get_logo_html(40)}<h1>Texo</h1></div><p>Lightweight LaTeX OCR · {model_wrapper.device_status}</p></div>')
        
        with gr.Row():
            # Left Column: Input
            with gr.Column():
                img_input = gr.Image(label="Upload Image", type="pil", height=300, sources=["upload", "clipboard"])
                recognize_btn = gr.Button("Recognize", variant="primary", size="lg", icon=str(LOGO_FILE))
                gr.Examples([[p] for p in EXAMPLE_IMAGES], inputs=img_input, label="Examples")
            
            # Right Column: Output
            with gr.Column():
                latex_output = gr.Textbox(
                    label="LaTeX", 
                    lines=6, 
                    max_lines=6, 
                    show_copy_button=True, 
                    elem_classes=["output-box"], 
                    elem_id="latex-output", 
                    interactive=True, 
                    placeholder="Result..."
                )
                
                conversion_result = gr.Textbox(
                    label="Conversion Result", 
                    visible=True, 
                    lines=6, 
                    max_lines=6, 
                    show_copy_button=True, 
                    elem_id="conversion-result"
                )
                
                # Hidden components for JS-Python communication
                hidden_format = gr.Textbox(elem_id="hidden-format", elem_classes=["hidden-box"], visible=True)
                hidden_convert_btn = gr.Button(elem_id="hidden-convert-btn", elem_classes=["hidden-box"], visible=True)
                
                # Custom Dropdown HTML
                gr.HTML(elem_classes=["dropdown-container"], value=DROPDOWN_HTML)
                
                # Preview
                preview_output = gr.Markdown(elem_classes=["preview-box"])
        
        # Footer
        gr.HTML(f'<div class="footer">{get_logo_html(16)} <a href="https://github.com/alephpi/Texo">GitHub</a> · AlephPi</div>')

        # --- Callbacks ---

        def run_recognition(image) -> Tuple[str, str]:
            latex = model_wrapper.predict(image)
            return latex, "" # Clear conversion result

        def perform_conversion(latex, action) -> str:
            """Handle conversion requests from the dropdown."""
            if not latex:
                return ""
            
            # Future feature implementations
            if action == "word":
                return convert_latex_to_mathml(latex)
            elif action == "ascii":
                return "AsciiMath not supported yet"
            elif action == "typst":
                return "Typst not supported yet"
            elif action == "docx":
                return "Docx export not supported yet"
            
            return f"Unknown action: {action}"

        # Event bindings
        recognize_btn.click(run_recognition, img_input, [latex_output, conversion_result])
        img_input.change(run_recognition, img_input, [latex_output, conversion_result])
        
        latex_output.change(format_latex_for_display, latex_output, preview_output)
        # Clear conversion result when latex changes manually
        latex_output.change(lambda: "", None, conversion_result)
        
        # Hidden button logic for dropdown actions
        hidden_convert_btn.click(
            perform_conversion,
            inputs=[latex_output, hidden_format],
            outputs=[conversion_result]
        ).then(
            None,
            [conversion_result, hidden_format],
            js="(content, action) => handleConversionResult(content, action)"
        )

    return demo

def main() -> None:
    print("=" * 40 + "\n  Texo - LaTeX OCR\n" + "=" * 40)
    
    model_wrapper = TexoModelWrapper()
    if not model_wrapper.load_model():
        sys.exit(1)
        
    try:
        ui = build_ui(model_wrapper)
        ui.launch(
            server_name="127.0.0.1",
            server_port=7860,
            inbrowser=True,
            show_error=True,
            share=False
        )
    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.exception("Application error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
