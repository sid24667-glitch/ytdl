"""Entry point of the YouTube content downloader.

Running ``python main.py`` opens the Tk window defined in :mod:`ui`.  The only
start-up work done here is a friendly pre-flight check for ``yt-dlp``: without
it the application cannot fetch anything, so instead of a traceback the user
sees an install hint (and the GUI still opens to show the same message).
"""

from __future__ import annotations

import sys


def _missing_dependencies() -> list[str]:
    """Return the names of required third-party modules that are unavailable."""
    missing: list[str] = []
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        missing.append("yt-dlp")
    try:
        import tkinter  # noqa: F401
    except ImportError:
        missing.append("tkinter (пакет python3-tk / Tcl-Tk)")
    return missing


def main() -> int:
    """Launch the graphical application; return a process exit code."""
    missing = _missing_dependencies()
    if missing:
        sys.stderr.write(
            "Не установлены зависимости: " + ", ".join(missing) + "\n"
            "Установите их командой: pip install -r requirements.txt\n"
        )
        return 1

    from ui import run

    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
