"""Texo - LaTeX OCR 图形界面."""
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

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

import gradio as gr
import torch
import latex2mathml.converter
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel

# Register custom model architecture
try:
    import texo.utils.config
except ImportError:
    # If src is not in path, try adding it
    sys.path.append(str(Path(__file__).parent / "src"))
    import texo.utils.config

from texo.data.processor import EvalMERImageProcessor

# 配置
MODEL_PATH = Path("./model")
LOGO_PATH = Path("assets/svg/logo.svg")
IMAGE_SIZE = {"width": 384, "height": 384}
EXAMPLE_IMAGES = [
    "./TechnoSelection/test_img/单行公式.png",
    "./TechnoSelection/test_img/单行公式2.png",
    "./TechnoSelection/test_img/多行公式.png",
    "./TechnoSelection/test_img/多行公式2.jpg",
]

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class _AsyncioFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(x in msg for x in ("ConnectionResetError", "_call_connection_lost", "WinError"))

logging.getLogger("asyncio").addFilter(_AsyncioFilter())

# CSS
CSS = """
.header { text-align: center; padding: 1.5rem; }
.header-title { display: flex; align-items: center; justify-content: center; gap: 0.5rem; }
.header h1 { margin: 0; font-size: 2rem; }
.header p { margin: 0.5rem 0 0; color: #666; }
.output-box textarea { font-family: Consolas, Monaco, monospace !important; max-height: 200px !important; overflow-y: auto !important; }
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

TOAST_JS = """
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

function getElementValue(id) {
    const el = document.querySelector(`#${id} textarea`);
    return el ? el.value : "";
}

function copyLatex(format) {
    const latex = getElementValue('latex-output');
    if (!latex) { showToast("没有可复制的内容"); return; }
    
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
    
    const resultBox = document.querySelector('#conversion-result textarea');
    if (resultBox) {
        resultBox.value = text;
        resultBox.dispatchEvent(new Event('input', { bubbles: true }));
    }
    
    navigator.clipboard.writeText(text).then(
        () => { showToast("已复制 LaTeX"); },
        (err) => { showToast("复制失败: " + err); }
    );
}

function copyMathML(action) {
    if (action === "word") {
        const mathml = getElementValue('mathml-storage');
        if (!mathml) { showToast("没有可复制的内容"); return; }
        
        const resultBox = document.querySelector('#conversion-result textarea');
        if (resultBox) {
            resultBox.value = mathml;
            resultBox.dispatchEvent(new Event('input', { bubbles: true }));
        }

        const blob = new Blob([mathml], {type: 'text/html'});
        const item = new ClipboardItem({'text/html': blob});
        navigator.clipboard.write([item]).then(
            () => { showToast("已复制 MathML，请在 Word 中直接粘贴"); },
            (err) => { showToast("复制失败: " + err); }
        );
    } else if (action === "ascii") {
        showToast("暂不支持 AsciiMath");
    } else if (action === "typst") {
        showToast("暂不支持 Typst");
    } else if (action === "docx") {
        showToast("请先安装 python-docx 库以支持导出功能");
    }
}
</script>
"""

@lru_cache(maxsize=1)
def read_logo() -> str:
    if not LOGO_PATH.exists():
        return ""
    return LOGO_PATH.read_text(encoding="utf-8")

def logo_html(size: int) -> str:
    svg = read_logo()
    if not svg:
        return ""
    svg = svg.replace("<svg ", '<svg width="100%" height="100%" ', 1)
    return f'<span style="display:inline-block;width:{size}px;height:{size}px;vertical-align:middle;">{svg}</span>'

def favicon() -> str:
    svg = read_logo()
    if not svg:
        return ""
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'

def standardize_latex(text: str) -> str:
    """标准化 LaTeX 命令 (修复 MathML 转换问题)."""
    # 使用正则确保只替换完整的命令 (避免 \left 被替换为 \leqft)
    def replace_command(text, old_cmd, new_cmd):
        # 匹配 \old_cmd 且后面不跟字母
        pattern = re.compile(re.escape(old_cmd) + r"(?![a-zA-Z])")
        # 使用 lambda 避免 new_cmd 中的反斜杠被 re.sub 当作转义符处理
        return pattern.sub(lambda m: new_cmd, text)

    replacements = [
        # 极限与算子
        (r"\operatorname*{lim}", r"\lim"),
        (r"\operatorname*{min}", r"\min"),
        (r"\operatorname*{max}", r"\max"),
        (r"\operatorname*{sup}", r"\sup"),
        (r"\operatorname*{inf}", r"\inf"),
        (r"\operatorname{lim}", r"\lim"),
        (r"\operatorname{min}", r"\min"),
        (r"\operatorname{max}", r"\max"),
        (r"\operatorname{sup}", r"\sup"),
        (r"\operatorname{inf}", r"\inf"),
        
        # 箭头与关系符
        (r"\rarr", r"\to"),
        (r"\infin", r"\infty"),
        (r"\ge", r"\geq"),
        (r"\le", r"\leq"),
        (r"\gt", ">"),
        (r"\lt", "<"),
        
        # 运算符号
        (r"\cdotp", r"\cdot"),
        (r"\lvert", "|"),
        (r"\rvert", "|"),
        (r"\lVert", "||"),
        (r"\rVert", "||"),
        
        # 字体与修饰
        (r"\Bbb", r"\mathbb"),
        (r"\bold", r"\mathbf"),
        (r"\rm", r"\mathrm"),
        (r"\it", r"\mathit"),
        (r"\bf", r"\mathbf"),
    ]
    
    for old, new in replacements:
        text = replace_command(text, old, new)

    text = text.replace("~", r"\quad")
    
    # 额外修复：移除 \operatorname* 中的 * (如果未被上述规则捕获)
    text = re.sub(r"\\operatorname\*\s*\{", r"\\operatorname{", text)
    
    # 修复：移除 \Im 和 \Re 前可能出现的错误竖线
    text = re.sub(r"\|\s*\\Im", r"\\Im", text)
    text = re.sub(r"\|\s*\\Re", r"\\Re", text)
    
    # 后处理：移除占位符
    text = re.sub(r"\\(black)?square", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\Box", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\boxed\s*\{\s*\}", "", text)
    text = re.sub(r"\\phantom\s*\{[^}]*\}", "", text)
    
    return text

def safe_convert_mathml(latex: str) -> str:
    """安全地将 LaTeX 转换为 MathML."""
    if not latex or latex.startswith("⚠️") or latex.startswith("("):
        return ""
    try:
        # 先标准化
        latex = standardize_latex(latex)
        return latex2mathml.converter.convert(latex)
    except Exception as e:
        logger.warning(f"MathML conversion failed: {e}")
        return ""

class TexoApp:
    """LaTeX OCR 应用."""

    def __init__(self) -> None:
        self.model: Optional[VisionEncoderDecoderModel] = None
        self.tokenizer = None
        self.processor: Optional[EvalMERImageProcessor] = None
        self.device: Optional[torch.device] = None
        self._error: Optional[str] = None
        atexit.register(self._cleanup)

    def _cleanup(self) -> None:
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self) -> bool:
        if self.model:
            return True
        if not MODEL_PATH.exists():
            self._error = f"模型不存在: {MODEL_PATH}"
            logger.error(self._error)
            return False
        try:
            start = time.time()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("设备: %s", self.device)
            self.model = VisionEncoderDecoderModel.from_pretrained(str(MODEL_PATH)).to(self.device).eval()  # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
            self.processor = EvalMERImageProcessor(image_size=IMAGE_SIZE)
            logger.info("加载完成 (%.2fs)", time.time() - start)
            return True
        except Exception as e:
            self._error = str(e)
            logger.exception("加载失败: %s", e)
            self._cleanup()
            return False

    def recognize(self, image: Optional[Image.Image]) -> str:
        if not self.model:
            return f"⚠️ {self._error or '模型未加载'}"
        if not image:
            return ""
        try:
            # 图像预处理：处理透明背景并转换为 RGB
            if image.mode != "RGB":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                    image = image.convert("RGBA")
                    bg.paste(image, mask=image.split()[-1])
                    image = bg
                else:
                    image = image.convert("RGB")
            
            # 推理
            x = self.processor(image).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                out = self.model.generate(pixel_values=x)
            
            result = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()  # type: ignore
            
            # 后处理：清理多余空格
            # 1. 保护 "\command letter" 格式的空格 (如 \sin x, \hat a)
            result = re.sub(r"(\\[a-zA-Z]+)\s+([a-zA-Z])", r"\1_TEXO_SP_\2", result)
            # 2. 移除所有其他空格
            result = result.replace(" ", "")
            # 3. 恢复保护的空格
            result = result.replace("_TEXO_SP_", " ")
            
            return result or "(未识别到内容)"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return "⚠️ 显存不足"
        except Exception as e:
            logger.exception("识别错误: %s", e)
            return f"⚠️ {e}"

    @property
    def status(self) -> str:
        if not self.model:
            return f"错误: {self._error[:30]}..." if self._error else "未加载"
        name = "GPU" if self.device.type == "cuda" else "CPU"  # type: ignore
        if self.device.type == "cuda":  # type: ignore
            name += f": {torch.cuda.get_device_name(0)}"
        return f"就绪 ({name})"

def render_latex(latex: str) -> str:
    if not latex or latex.startswith("⚠️") or latex.startswith("("):
        return f"*{latex}*" if latex else ""
    return f"$$\n{latex.strip()}\n$$"

def create_ui(app: TexoApp) -> gr.Blocks:
    with gr.Blocks(title="Texo", css=CSS, head=favicon() + TOAST_JS) as demo:
        gr.HTML(f'<div class="header"><div class="header-title">{logo_html(40)}<h1>Texo</h1></div><p>轻量级 LaTeX OCR · {app.status}</p></div>')
        
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="上传图片", type="pil", height=300, sources=["upload", "clipboard"])
                btn = gr.Button("识别", variant="primary", size="lg", icon=str(LOGO_PATH))
                gr.Examples([[p] for p in EXAMPLE_IMAGES], inputs=img, label="示例")
            
            with gr.Column():
                out = gr.Textbox(label="LaTeX", lines=6, max_lines=6, show_copy_button=True, elem_classes=["output-box"], elem_id="latex-output", interactive=True, placeholder="识别结果...")
                conversion_result = gr.Textbox(label="转换结果", visible=True, lines=4, max_lines=4, show_copy_button=True, elem_id="conversion-result")
                mathml_storage = gr.Textbox(elem_id="mathml-storage", elem_classes=["hidden-box"], visible=True)
                
                gr.HTML(elem_classes=["dropdown-container"], value="""
                <div style="display: flex; gap: 10px; width: 100%;">
                    <div class="custom-dropdown">
                        <div class="custom-dropbtn">复制 LaTeX <span style="font-size: 10px;">▼</span></div>
                        <div class="custom-dropdown-content">
                            <div onclick="copyLatex('raw')">无特殊附加</div>
                            <div onclick="copyLatex('$')">$ ... $ 格式</div>
                            <div onclick="copyLatex('$$')">$$ ... $$ 格式</div>
                            <div onclick="copyLatex('\\[')">\\[ ... \\] 格式</div>
                            <div onclick="copyLatex('\\(')">\\( ... \\) 格式</div>
                            <div onclick="copyLatex('equation')">\\begin{equation} ... \\end{equation} 格式</div>
                        </div>
                    </div>
                    <div class="custom-dropdown">
                        <div class="custom-dropbtn">复制 MathML <span style="font-size: 10px;">▼</span></div>
                        <div class="custom-dropdown-content">
                            <div onclick="copyMathML('word')">复制MathML(Word)</div>
                            <div onclick="copyMathML('ascii')">复制AsciiMath</div>
                            <div onclick="copyMathML('typst')">复制Typst</div>
                            <div onclick="copyMathML('docx')">导出Docx(Word/WPS)</div>
                        </div>
                    </div>
                </div>
                """)
                
                preview = gr.Markdown(elem_classes=["preview-box"])
        
        gr.HTML(f'<div class="footer">{logo_html(16)} <a href="https://github.com/alephpi/Texo">GitHub</a> · AlephPi</div>')

        def recognize_and_convert(image) -> Tuple[str, str, str]:
            latex = app.recognize(image)
            mathml = safe_convert_mathml(latex)
            return latex, mathml, ""

        def update_conversion_data(latex) -> Tuple[str, str]:
            mathml = safe_convert_mathml(latex)
            return mathml, ""

        # 事件绑定
        btn.click(recognize_and_convert, img, [out, mathml_storage, conversion_result])
        img.change(recognize_and_convert, img, [out, mathml_storage, conversion_result])
        out.change(render_latex, out, preview)
        out.change(update_conversion_data, out, [mathml_storage, conversion_result])

    return demo


def main() -> None:
    print("=" * 40 + "\n  Texo - LaTeX OCR\n" + "=" * 40)
    app = TexoApp()
    if not app.load():
        sys.exit(1)
    try:
        create_ui(app).launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, show_error=True)
    except KeyboardInterrupt:
        logger.info("退出")
    except Exception as e:
        logger.exception("错误: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
