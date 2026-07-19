"""Entrypoint for the RMS Photo Processor service.

Run from source:  python run.py [--port 8765]
Frozen exe:       photo-processor.exe [--port 8765] [--install-startup]
"""

import argparse
import logging
import os
import subprocess
import sys

APP_NAME = "RMS Photo Processor"
# Custom URL scheme so the RMS web app can launch an installed exe on demand
# (browser navigates to rms-photoprocessor://start). Registered in HKCU on
# every packaged run, so moving the exe self-heals on next manual launch.
PROTOCOL = "rms-photoprocessor"


def _configure_model_cache() -> None:
    """Point rembg's downloader at our local cache before it is imported."""
    from app import config

    os.environ.setdefault("U2NET_HOME", config.cache_dir())


def _startup_shortcut_path() -> str:
    return os.path.join(
        os.environ["APPDATA"],
        "Microsoft", "Windows", "Start Menu", "Programs", "Startup",
        f"{APP_NAME}.lnk",
    )


def install_startup() -> None:
    if not getattr(sys, "frozen", False):
        print("Startup shortcut is only supported for the packaged .exe")
        return
    lnk = _startup_shortcut_path()
    target = sys.executable
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{lnk}'); "
        "$s.TargetPath = '{target}'; "
        "$s.WorkingDirectory = '{cwd}'; "
        "$s.WindowStyle = 7; "
        "$s.Save()"
    ).format(lnk=lnk, target=target, cwd=os.path.dirname(target))
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
    )
    print(f"Startup shortcut created: {lnk}")


def register_protocol() -> None:
    """Best-effort HKCU registration of rms-photoprocessor:// → this exe."""
    if not getattr(sys, "frozen", False):
        return
    try:
        import winreg

        base = rf"Software\Classes\{PROTOCOL}"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as key:
            winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f"URL:{APP_NAME}")
            winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
        with winreg.CreateKey(
            winreg.HKEY_CURRENT_USER, base + r"\shell\open\command"
        ) as key:
            winreg.SetValueEx(
                key, None, 0, winreg.REG_SZ, f'"{sys.executable}" "%1"'
            )
    except OSError as exc:
        print(f"Could not register {PROTOCOL}:// protocol: {exc}")


def port_in_use(host: str, port: int) -> bool:
    import socket

    with socket.socket() as sock:
        return sock.connect_ex((host, port)) == 0


def uninstall_startup() -> None:
    lnk = _startup_shortcut_path()
    if os.path.exists(lnk):
        os.remove(lnk)
        print(f"Removed {lnk}")
    else:
        print("No startup shortcut installed")


def main() -> None:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--port", type=int, default=None, help="Port (default 8765)")
    parser.add_argument("--host", default="127.0.0.1", help=argparse.SUPPRESS)
    parser.add_argument(
        "--install-startup", action="store_true",
        help="Create a Windows startup shortcut for this exe and exit",
    )
    parser.add_argument(
        "--uninstall-startup", action="store_true",
        help="Remove the Windows startup shortcut and exit",
    )
    # When launched via rms-photoprocessor:// the browser passes the URL as an
    # argument; accept and ignore it.
    parser.add_argument("launch_url", nargs="?", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.install_startup:
        install_startup()
        return
    if args.uninstall_startup:
        uninstall_startup()
        return

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _configure_model_cache()

    from app import config
    from app.main import app, start_warmup

    port = args.port if args.port is not None else config.port()
    if port_in_use(args.host, port):
        # On-demand launches from the web app land here when the service is
        # already up — that's success, not an error.
        print(f"{APP_NAME} is already running on port {port}.")
        return
    register_protocol()
    start_warmup()

    import uvicorn

    print(f"{APP_NAME} v{config.VERSION} — http://{args.host}:{port}")
    print(f"Model: {config.model_name()} ({config.model_license()}), "
          f"cache: {config.cache_dir()}")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    # PyInstaller onefile: child processes re-exec the bootloader; guard keeps
    # multiprocessing-safe even though we run single-process.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
