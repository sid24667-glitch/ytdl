"""Persistent user configuration stored next to the application.

Only a handful of values have to survive between runs (the last download
directory and the last chosen format), so the module keeps the JSON file
tiny and tolerant: unknown/missing keys fall back to defaults instead of
raising, which is important because the file may be edited by hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_FILE_NAME = "config.json"


@dataclass
class Config:
    """Serializable application settings."""

    download_dir: str = ""
    last_format_token: str = ""
    last_url: str = ""

    @property
    def resolved_download_dir(self) -> str:
        """Download directory, defaulting to the user's ``Downloads`` folder."""
        if self.download_dir and os.path.isdir(self.download_dir):
            return self.download_dir
        home = Path.home()
        for candidate in ("Downloads", "Загрузки"):
            path = home / candidate
            if path.is_dir():
                return str(path)
        return str(home)


def config_path(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Absolute path of the configuration file."""
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    return root / CONFIG_FILE_NAME


def load_config(base_dir: str | os.PathLike[str] | None = None) -> Config:
    """Read the configuration; a missing or broken file yields defaults."""
    path = config_path(base_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Config()
    if not isinstance(raw, dict):
        return Config()
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in known})


def save_config(config: Config, base_dir: str | os.PathLike[str] | None = None) -> bool:
    """Persist *config*; returns ``True`` on success (errors are swallowed)."""
    try:
        config_path(base_dir).write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True
