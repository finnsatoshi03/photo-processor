# RMS Photo Processor

Local Windows service that prepares ID photos for the RMS Printing page:
background removal → solid background color → face-centered auto-crop →
exact-pixel resize at 300 DPI. The RMS web app (Vercel) calls it over
`http://127.0.0.1:8765` from the same PC. API: see [../CONTRACT.md](../CONTRACT.md).

- **Model**: rembg `u2net` (Apache-2.0) by default; `birefnet-general` (MIT)
  selectable via `PHOTO_PROCESSOR_MODEL=birefnet-general` (better edges,
  much slower on CPU). Non-commercial models (BRIA RMBG) are rejected.
- **Face detection**: OpenCV YuNet (MIT model, downloaded to cache) with a
  bundled Haar-cascade fallback for offline first runs.
- **Models cache**: `%LOCALAPPDATA%\PhotoProcessor\models` — downloaded on
  first run, reused afterwards.

## Run from source (dev)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python run.py
```

## Build the shop .exe

```powershell
.\build.ps1          # CPU
.\build.ps1 -Gpu     # NVIDIA GPU (onnxruntime-gpu)
```

Produces `dist\photo-processor.exe` (one file, no Python needed on the shop
PC). Double-click to run; the console window shows the status. First launch
downloads the model, `/health` reports `"loading"` until it is ready.

```powershell
photo-processor.exe --install-startup    # start automatically with Windows
photo-processor.exe --uninstall-startup
photo-processor.exe --port 9000          # non-default port
```

## On-demand launch from RMS

Every packaged run registers the `rms-photoprocessor://` URL scheme (HKCU,
no admin rights) pointing at the exe's current location. The RMS Printing
page uses it to start the processor automatically when it isn't running.
So the shop setup is: download the exe (GitHub release), run it once
(model downloads, protocol registers) — after that the web app launches it
on demand. Starting a second instance while one is running just prints a
notice and exits.

## Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `PHOTO_PROCESSOR_PORT` | `8765` | Listen port (contract default). |
| `PHOTO_PROCESSOR_MODEL` | `u2net` | `u2net` or `birefnet-general`. |
| `PHOTO_PROCESSOR_ORIGINS` | – | Extra allowed CORS origins (comma-separated). `https://*.vercel.app` and Vite dev origins are always allowed. |

## Smoke test

```powershell
curl http://127.0.0.1:8765/health
curl -F "image=@photo.jpg" -F "size_preset=2x2" -F "bg_color=#2e6db4" http://127.0.0.1:8765/process
```
