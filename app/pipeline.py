"""Image pipeline: background removal -> face-centered crop -> exact-size PNG.

Order matters: the face is detected on the original photo, the crop box is
computed at the target aspect ratio, and the cutout is composited onto the
solid background at crop time — so a crop box that extends past the photo's
edges simply gains seamless background instead of black bars.
"""

import io
import logging
import threading

import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

from . import config, faces
from .presets import DPI

log = logging.getLogger("photo-processor")

# ID-photo composition: the head (crown to chin) fills ~70% of the frame height
# with ~12% clear space above the crown — matches PH passport/visa guidance.
HEAD_FRACTION = 0.70
TOP_MARGIN_FRACTION = 0.12
# Face detectors box roughly eyebrows-to-chin; expand to estimate the full head.
CROWN_EXPAND = 0.45
CHIN_EXPAND = 0.10

_session = None
_session_lock = threading.Lock()


def warmup() -> None:
    """Build the rembg session and run a tiny image through it. Called from a
    background thread at startup; /process waits on readiness via main.py."""
    global _session
    with _session_lock:
        if _session is None:
            _session = new_session(config.model_name())
    faces.prepare()
    tiny = Image.new("RGB", (32, 32), (128, 128, 128))
    remove(tiny, session=_session)
    log.info("Model %s ready", config.model_name())


def gpu_active() -> bool:
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        return any(p in providers for p in ("CUDAExecutionProvider", "DmlExecutionProvider"))
    except Exception:
        return False


def parse_hex_color(value: str) -> tuple[int, int, int]:
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bg_color must be #RGB or #RRGGBB, got {value!r}")
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        raise ValueError(f"bg_color must be hex, got {value!r}") from None


def process_image(
    image_bytes: bytes,
    width_px: int,
    height_px: int,
    bg_rgb: tuple[int, int, int],
    auto_crop: bool,
) -> tuple[bytes, bool]:
    """Return (png_bytes, face_detected)."""
    try:
        src = Image.open(io.BytesIO(image_bytes))
        src = ImageOps.exif_transpose(src).convert("RGB")
    except Exception:
        raise ValueError("Could not decode image") from None

    cutout = remove(src, session=_session)  # RGBA, background transparent

    aspect = width_px / height_px
    box = None
    face_detected = False
    if auto_crop:
        box = _face_crop_box(np.asarray(src), aspect)
        face_detected = box is not None
    if box is None:
        box = _center_cover_box(src.width, src.height, aspect)

    left, top, crop_w, crop_h = box
    canvas = Image.new("RGB", (round(crop_w), round(crop_h)), bg_rgb)
    canvas.paste(cutout, (-round(left), -round(top)), cutout)

    out = canvas.resize((width_px, height_px), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG", dpi=(DPI, DPI))
    return buf.getvalue(), face_detected


def _face_crop_box(
    rgb: np.ndarray, aspect: float
) -> tuple[float, float, float, float] | None:
    """Crop box (left, top, w, h) centering the head, may exceed image bounds."""
    face = faces.detect_face(rgb)
    if face is None:
        return None
    fx, fy, fw, fh = face
    head_top = fy - CROWN_EXPAND * fh
    head_h = fh * (1 + CROWN_EXPAND + CHIN_EXPAND)
    crop_h = head_h / HEAD_FRACTION
    top = head_top - TOP_MARGIN_FRACTION * crop_h
    crop_w = crop_h * aspect
    left = (fx + fw / 2) - crop_w / 2
    return left, top, crop_w, crop_h


def _center_cover_box(
    img_w: int, img_h: int, aspect: float
) -> tuple[float, float, float, float]:
    if img_w / img_h > aspect:
        crop_h = float(img_h)
        crop_w = crop_h * aspect
    else:
        crop_w = float(img_w)
        crop_h = crop_w / aspect
    return (img_w - crop_w) / 2, (img_h - crop_h) / 2, crop_w, crop_h
