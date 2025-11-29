"""Texo - LaTeX OCR 图形界面."""
import atexit
import base64
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "16")

import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel

from texo.data.processor import EvalMERImageProcessor
from texo.utils.config import AutoConfig, AutoModel, HGNetv2, HGNetv2Config  # noqa: F401

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

# 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 过滤 Windows asyncio 噪音日志
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
"""


@lru_cache(maxsize=1)
def read_logo() -> str:
    return LOGO_PATH.read_text(encoding="utf-8")


def logo_html(size: int) -> str:
    svg = read_logo().replace("<svg ", '<svg width="100%" height="100%" ', 1)
    return f'<span style="display:inline-block;width:{size}px;height:{size}px;vertical-align:middle;">{svg}</span>'


def favicon() -> str:
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{base64.b64encode(read_logo().encode()).decode()}">'


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
            # 转 RGB
            if image.mode != "RGB":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                if image.mode in ("RGBA", "LA", "P"):
                    if image.mode == "P":
                        image = image.convert("RGBA")
                    bg.paste(image, mask=image.split()[-1] if image.mode in ("RGBA", "LA") else None)
                    image = bg
                else:
                    image = image.convert("RGB")
            # 推理
            x = self.processor(image).unsqueeze(0).to(self.device)  # type: ignore
            with torch.no_grad():
                out = self.model.generate(pixel_values=x)
            result = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()  # type: ignore
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
    with gr.Blocks(title="Texo", css=CSS, head=favicon()) as demo:
        gr.HTML(f'<div class="header"><div class="header-title">{logo_html(40)}<h1>Texo</h1></div><p>轻量级 LaTeX OCR · {app.status}</p></div>')
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="上传图片", type="pil", height=300, sources=["upload", "clipboard"])
                btn = gr.Button("识别", variant="primary", size="lg", icon=str(LOGO_PATH))
                gr.Examples([[p] for p in EXAMPLE_IMAGES], inputs=img, label="示例")
            with gr.Column():
                out = gr.Textbox(label="LaTeX", lines=6, max_lines=6, show_copy_button=True, elem_classes=["output-box"], interactive=True, placeholder="识别结果...")
                preview = gr.Markdown(elem_classes=["preview-box"])
        gr.HTML(f'<div class="footer">{logo_html(16)} <a href="https://github.com/alephpi/Texo">GitHub</a> · AlephPi</div>')
        btn.click(app.recognize, img, out)
        img.change(app.recognize, img, out)
        out.change(render_latex, out, preview)
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
