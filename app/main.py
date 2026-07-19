"""FastAPI app implementing ../CONTRACT.md."""

import base64
import binascii
import logging
import threading

import anyio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config, formdata, pipeline
from .presets import resolve_size

log = logging.getLogger("photo-processor")

app = FastAPI(title="RMS Photo Processor", version=config.VERSION, docs_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.PROD_ORIGINS + config.DEV_ORIGINS + config.extra_origins(),
    allow_origin_regex=config.ORIGIN_REGEX,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,
)

_ready = threading.Event()
_warmup_error: str | None = None


@app.middleware("http")
async def private_network_preflight(request: Request, call_next):
    """Legacy Chromium Private Network Access: echo the allow header on
    preflight. Harmless under the newer permission-prompt LNA model."""
    response = await call_next(request)
    if (
        request.method == "OPTIONS"
        and request.headers.get("access-control-request-private-network") == "true"
    ):
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


def start_warmup() -> None:
    def run() -> None:
        global _warmup_error
        try:
            pipeline.warmup()
            _ready.set()
        except Exception as exc:
            _warmup_error = str(exc)
            log.exception("Model warmup failed")

    threading.Thread(target=run, name="warmup", daemon=True).start()


@app.get("/health")
def health():
    return {
        "status": "ok" if _ready.is_set() else "loading",
        "version": config.VERSION,
        "model": config.model_name(),
        "license": config.model_license(),
        "gpu": pipeline.gpu_active(),
        # Per-model readiness so clients can offer a model picker and show
        # download progress states: unloaded | loading | ready.
        "models": pipeline.model_states(),
        # {"<model>": {"done": bytes, "total": approx bytes}} while loading.
        "model_progress": pipeline.model_progress(),
    }


@app.post("/models/warm")
async def warm_model(request: Request):
    """Start loading a model in the background (idempotent). The client polls
    /health until the model reports ready."""
    try:
        body = await request.json()
        model = body.get("model") if isinstance(body, dict) else None
    except Exception:
        model = None
    if model not in config.ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="model must be one of: " + ", ".join(sorted(config.ALLOWED_MODELS)),
        )
    return {"model": model, "state": pipeline.ensure_model(model)}


@app.post("/process")
async def process(request: Request):
    if not _ready.is_set():
        detail = "Model is still loading; retry shortly"
        if _warmup_error:
            detail = f"Model failed to load: {_warmup_error}"
        raise HTTPException(status_code=503, detail=detail)

    body = await request.body()
    if len(body) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 25 MB")

    content_type = request.headers.get("content-type", "")
    try:
        if content_type.startswith("multipart/form-data"):
            fields = formdata.parse_multipart(content_type, body)
            image_bytes = fields.get("image")
            if not isinstance(image_bytes, bytes):
                raise ValueError("Missing file field 'image'")
        elif content_type.startswith("application/json"):
            import json

            fields = json.loads(body)
            if not isinstance(fields, dict):
                raise ValueError("JSON body must be an object")
            image_bytes = _decode_base64_image(fields.get("image"))
        else:
            raise ValueError(
                "Content-Type must be multipart/form-data or application/json"
            )

        width_px, height_px = resolve_size(
            _opt_str(fields.get("size_preset")),
            _opt_int(fields.get("width_px"), "width_px"),
            _opt_int(fields.get("height_px"), "height_px"),
        )
        bg_rgb = pipeline.parse_hex_color(_opt_str(fields.get("bg_color")) or "#FFFFFF")
        auto_crop = _opt_bool(fields.get("auto_crop"), default=True)
        model = _opt_str(fields.get("model")) or config.model_name()
        if model not in config.ALLOWED_MODELS:
            raise ValueError(
                "model must be one of: " + ", ".join(sorted(config.ALLOWED_MODELS))
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    state = pipeline.ensure_model(model)
    if state != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model {model} is still downloading/loading; retry shortly",
        )

    try:
        png, face_detected, face_box = await anyio.to_thread.run_sync(
            pipeline.process_image,
            image_bytes,
            width_px,
            height_px,
            bg_rgb,
            auto_crop,
            model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        log.exception("Processing failed")
        raise HTTPException(status_code=500, detail="Processing failed") from None

    return {
        "image": base64.b64encode(png).decode("ascii"),
        "width_px": width_px,
        "height_px": height_px,
        "face_detected": face_detected,
        "face_box": face_box,
    }


def _decode_base64_image(value) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing base64 field 'image'")
    if value.startswith("data:"):  # tolerated per contract
        _, _, value = value.partition(",")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Field 'image' is not valid base64") from None


def _opt_str(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected a string value")
    return value or None


def _opt_int(value, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None


def _opt_bool(value, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    raise ValueError("auto_crop must be a boolean")
