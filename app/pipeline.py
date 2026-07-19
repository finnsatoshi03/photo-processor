"""Image pipeline: background removal -> face-centered crop -> exact-size PNG.

Order matters: the face is detected on the original photo, the crop box is
computed at the target aspect ratio, and the cutout is composited onto the
solid background at crop time — so a crop box that extends past the photo's
edges simply gains seamless background instead of black bars.
"""

import io
import logging
import os
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

# One rembg session per model, loaded on demand so the heavyweight
# birefnet-general (~1 GB download) costs nothing until a user picks it.
_state_lock = threading.Lock()
_sessions: dict[str, object] = {}
_states: dict[str, str] = {name: "unloaded" for name in config.ALLOWED_MODELS}
# Download progress per loading model: {"done": bytes, "total": bytes}.
_progress: dict[str, dict[str, int]] = {}

# rembg's own model sources (its pooch downloader hides progress in a temp
# file, so we fetch the same files ourselves and report exact byte counts;
# rembg then finds them in the cache and skips its download).
MODEL_URLS = {
    "u2net": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
    "birefnet-general": "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx",
}
MODEL_FILES = {"u2net": "u2net.onnx", "birefnet-general": "birefnet-general.onnx"}
# Fallback totals when the server sends no Content-Length.
MODEL_TOTALS = {"u2net": 176_000_000, "birefnet-general": 975_000_000}


def model_states() -> dict[str, str]:
    """Per-model state for /health: unloaded | loading | ready."""
    with _state_lock:
        return dict(_states)


def model_progress() -> dict[str, dict[str, int]]:
    """Download progress for models currently loading, for /health."""
    with _state_lock:
        return {k: dict(v) for k, v in _progress.items()}


def _predownload(name: str) -> None:
    """Fetch the model file with byte-accurate progress. Best-effort: on any
    failure rembg's own downloader takes over (without progress reporting)."""
    import urllib.request

    dest = os.path.join(config.cache_dir(), MODEL_FILES[name])
    if os.path.exists(dest):
        return
    tmp = dest + ".part"
    req = urllib.request.Request(
        MODEL_URLS[name], headers={"User-Agent": "rms-photo-processor"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0) or MODEL_TOTALS[name]
        done = 0
        while True:
            chunk = resp.read(256 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            with _state_lock:
                _progress[name] = {"done": done, "total": total}
    os.replace(tmp, dest)
    log.info("Downloaded %s (%d bytes)", MODEL_FILES[name], done)


def _load_model(name: str) -> None:
    try:
        try:
            _predownload(name)
        except Exception as exc:
            log.warning("Pre-download of %s failed (%s); rembg will retry", name, exc)
        finally:
            # Session init follows; the UI switches to its indeterminate
            # "preparing" message once progress disappears.
            with _state_lock:
                _progress.pop(name, None)
        session = new_session(name)
        tiny = Image.new("RGB", (32, 32), (128, 128, 128))
        remove(tiny, session=session)
        with _state_lock:
            _sessions[name] = session
            _states[name] = "ready"
        log.info("Model %s ready", name)
    except Exception:
        with _state_lock:
            _states[name] = "unloaded"
        log.exception("Loading model %s failed", name)
    finally:
        with _state_lock:
            _progress.pop(name, None)


def ensure_model(name: str) -> str:
    """Kick off a background load when needed; return the state right now."""
    with _state_lock:
        state = _states[name]
        if state != "unloaded":
            return state
        _states[name] = "loading"
    threading.Thread(
        target=_load_model, args=(name,), name=f"load-{name}", daemon=True
    ).start()
    return "loading"


def warmup() -> None:
    """Load the default model. Called from a background thread at startup;
    /process waits on readiness via main.py."""
    default = config.model_name()
    with _state_lock:
        _states[default] = "loading"
    faces.prepare()
    _load_model(default)
    if model_states()[default] != "ready":
        raise RuntimeError(f"default model {default} failed to load")


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
    model: str,
) -> tuple[bytes, bool]:
    """Return (png_bytes, face_detected). Caller guarantees `model` is ready."""
    with _state_lock:
        session = _sessions.get(model)
    if session is None:
        raise RuntimeError(f"model {model} not loaded")

    try:
        src = Image.open(io.BytesIO(image_bytes))
        src = ImageOps.exif_transpose(src).convert("RGB")
    except Exception:
        raise ValueError("Could not decode image") from None

    cutout = remove(src, session=session)  # RGBA, background transparent

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
