"""Face detection for auto-crop.

Primary: OpenCV YuNet (tiny ONNX model, MIT license) downloaded to the local
cache on first run. Fallback: OpenCV's bundled Haar cascade, which needs no
download, so the service still auto-crops when offline on first launch.
"""

import logging
import os
import urllib.request

import cv2
import numpy as np

from . import config

log = logging.getLogger("photo-processor")

# Detection runs on a downscaled copy; boxes are scaled back to source pixels.
DETECT_MAX_SIDE = 1024

_yunet_path: str | None = None


def prepare() -> None:
    """Best-effort YuNet download; called once at startup from the warmup thread."""
    global _yunet_path
    path = os.path.join(config.cache_dir(), config.YUNET_FILENAME)
    if not os.path.exists(path):
        try:
            log.info("Downloading YuNet face model to %s", path)
            tmp = path + ".part"
            urllib.request.urlretrieve(config.YUNET_URL, tmp)
            os.replace(tmp, path)
        except Exception as exc:  # offline first run: Haar fallback still works
            log.warning("YuNet download failed (%s); using Haar cascade", exc)
            return
    _yunet_path = path


def detect_face(rgb: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return the most confident face box (x, y, w, h) in source pixels, or None."""
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(w, h))
    small = (
        cv2.resize(rgb, (round(w * scale), round(h * scale))) if scale < 1.0 else rgb
    )
    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)

    box = _detect_yunet(bgr)
    if box is None:
        box = _detect_haar(bgr)
    if box is None:
        return None
    x, y, bw, bh = box
    return (x / scale, y / scale, bw / scale, bh / scale)


def _detect_yunet(bgr: np.ndarray) -> tuple[float, float, float, float] | None:
    if _yunet_path is None:
        return None
    try:
        h, w = bgr.shape[:2]
        det = cv2.FaceDetectorYN.create(_yunet_path, "", (w, h), 0.6, 0.3, 5000)
        _, faces = det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])
        return tuple(float(v) for v in best[:4])
    except cv2.error as exc:
        log.warning("YuNet detection failed (%s)", exc)
        return None


def _detect_haar(bgr: np.ndarray) -> tuple[float, float, float, float] | None:
    cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )
    if cascade.empty():
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    best = max(faces, key=lambda f: f[2] * f[3])
    return tuple(float(v) for v in best)
