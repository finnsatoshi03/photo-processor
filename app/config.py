"""Service configuration. Values here must stay in sync with ../CONTRACT.md."""

import os

VERSION = "1.2.1"
DEFAULT_PORT = 8765
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Commercial-safe background-removal models only (contract §health).
# bria-rmbg / RMBG is CC BY-NC (non-commercial) and must never be added here.
ALLOWED_MODELS = {
    "u2net": "Apache-2.0",
    "birefnet-general": "MIT",
}
DEFAULT_MODEL = "u2net"

DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
# The deployed RMS app. Additional origins: PHOTO_PROCESSOR_ORIGINS env var.
PROD_ORIGINS = [
    "https://rmsavisha.company",
    "https://www.rmsavisha.company",
]
# Vercel preview/default deployments are also accepted.
ORIGIN_REGEX = r"https://.*\.vercel\.app"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"


def model_name() -> str:
    name = os.environ.get("PHOTO_PROCESSOR_MODEL", DEFAULT_MODEL).strip()
    if name not in ALLOWED_MODELS:
        raise SystemExit(
            f"Unsupported model {name!r}. Allowed (commercial-safe) models: "
            + ", ".join(sorted(ALLOWED_MODELS))
        )
    return name


def model_license() -> str:
    return ALLOWED_MODELS[model_name()]


def cache_dir() -> str:
    """Local model cache; created on first run. Also handed to rembg via U2NET_HOME."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "PhotoProcessor", "models")
    os.makedirs(path, exist_ok=True)
    return path


def extra_origins() -> list[str]:
    raw = os.environ.get("PHOTO_PROCESSOR_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


def port() -> int:
    return int(os.environ.get("PHOTO_PROCESSOR_PORT", DEFAULT_PORT))
