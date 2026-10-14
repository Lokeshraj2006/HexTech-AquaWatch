"""Keep AquaWatch offline: no usage analytics, no telemetry, no surprise downloads (REAL_WORLD_CHECK G6).

    import offline
    offline.apply()             # before torch / Ultralytics / open_clip are imported (cpu_limits.apply() calls it)

Before 2026-10-12 only start_aquawatch.bat set these, so `python run.py` or `ingest.py` from a command line used the
user's own Ultralytics settings, which on this laptop have "sync": true (usage analytics on when online).
Each variable is only set when it is not set already, so a person who WANTS a download can still allow it, e.g.
`set YOLO_OFFLINE=false` before running.

    YOLO_OFFLINE=true              Ultralytics treats the computer as offline: analytics events off, no downloads
    YOLO_CONFIG_DIR=<folder above the repo>/.ultralytics   the project's own Ultralytics settings ("sync": false), not the user's
    HF_HUB_OFFLINE=1               Hugging Face hub (open_clip) reads only local files
    HF_HUB_DISABLE_TELEMETRY=1     ...and sends no telemetry
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".ultralytics"


def settings() -> dict:
    """The environment variables AquaWatch sets when they are missing."""
    return {"YOLO_OFFLINE": "true", "YOLO_CONFIG_DIR": str(PROJECT_CONFIG_DIR), "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}


def apply() -> dict:
    """Set every missing variable of settings(). Returns the ones this call set (empty when all were set already)."""
    changed = {}
    for key, value in settings().items():
        if key not in os.environ:
            os.environ[key] = value
            changed[key] = value
    return changed
