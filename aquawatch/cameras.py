"""Camera registry for AquaWatch (multi-camera pool CCTV).

A registry is a small YAML file that names every camera of one site:

    site: "AquaWatch demo pool"
    cameras:
      - id: cam_deck                      # [a-z0-9_]+, unique
        name: "Deck / Entry"              # human name, used in answers
        video: samples/multicam/cam_deck.mp4
        start_time: "2026-10-09 14:00:00" # wall clock of the first frame (local time)
        preset: pool                      # scenarios/<preset>.yaml
        zones: samples/multicam/cam_deck_zones.json   # optional
        offset_s: 0.0                     # optional: add to video seconds -> shared site timeline
        sync_status: user_declared        # optional: synchronized | user_declared | unavailable

Synchronization: `start_time` is only the clock this camera's recorder claims for its first frame.
Cross-camera time comparisons need an alignment, given by `offset_s` + `sync_status`:
  - "synchronized"  : the offset was measured (e.g. a shared flash / timecode);
  - "user_declared" : the user states the offset (not measured);
  - "unavailable"   : no alignment (the default). normalized_time() then returns (None, "unavailable")
                      and callers must say "Temporal alignment unavailable."
An `offset_s` without a `sync_status` counts as "user_declared" (the user wrote it in the registry).

Relative paths are relative to the project root (the folder of this file).
Pure Python (+ PyYAML); no cv2 / torch.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent   # the project folder
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
DEFAULT_PRESET = "pool"

# synchronization status of a camera (see the module docstring)
SYNC_SYNCHRONIZED = "synchronized"
SYNC_USER_DECLARED = "user_declared"
SYNC_UNAVAILABLE = "unavailable"
SYNC_STATUSES = (SYNC_SYNCHRONIZED, SYNC_USER_DECLARED, SYNC_UNAVAILABLE)
DEFAULT_SYNC_STATUS = SYNC_UNAVAILABLE
TEMPORAL_UNAVAILABLE = "Temporal alignment unavailable."

# the cameras of the registry loaded last, so camera_name(id) works without passing the list around
_LAST: list[dict] = []


def parse_wall(value: Any) -> datetime:
    """'2026-10-09 14:00:00' / ISO string / datetime -> naive datetime (local wall clock)."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if not text:
        raise ValueError("empty time")
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError as exc:
        raise ValueError(f"cannot read time {text!r} (use 'YYYY-MM-DD HH:MM:SS')") from exc


def format_wall(moment: datetime) -> str:
    """Fixed-width ISO text 'YYYY-MM-DDTHH:MM:SS.mmm' so wall times sort and compare as strings."""
    return moment.isoformat(timespec="milliseconds")


def normalize_wall(value: Any) -> str:
    """Any accepted time (text or datetime) -> the fixed ISO format used in the index."""
    return format_wall(parse_wall(value))


def resolve_path(value: Any, root: Path = ROOT) -> Path:
    """A registry path (relative to the project root, or absolute) -> absolute Path."""
    path = Path(str(value))
    return path if path.is_absolute() else (Path(root) / path)


def _warn(message: str, quiet: bool) -> None:
    """Print a plain ASCII warning to stderr (unless quiet)."""
    if not quiet:
        print(f"warning: {message}", file=sys.stderr)


def _camera(raw: Any, position: int, root: Path, scenarios_dir: Path, quiet: bool) -> dict:
    """Validate one registry entry and return the camera dict."""
    if not isinstance(raw, dict):
        raise ValueError(f"camera #{position}: expected a mapping, got {type(raw).__name__}")
    cam_id = str(raw.get("id") or "").strip()
    if not ID_PATTERN.match(cam_id):
        raise ValueError(f"camera #{position}: id {cam_id!r} must match [a-z0-9_]+")
    if not raw.get("video"):
        raise ValueError(f"camera {cam_id}: 'video' is missing")
    if raw.get("start_time") in (None, ""):
        raise ValueError(f"camera {cam_id}: 'start_time' is missing")
    try:
        start = parse_wall(raw["start_time"])
    except ValueError as exc:
        raise ValueError(f"camera {cam_id}: start_time: {exc}") from exc

    preset = str(raw.get("preset") or DEFAULT_PRESET).strip()
    if not (Path(scenarios_dir) / f"{preset}.yaml").exists():
        known = sorted(p.stem for p in Path(scenarios_dir).glob("*.yaml"))
        raise ValueError(f"camera {cam_id}: preset {preset!r} not in scenarios/ (known: {', '.join(known)})")

    warnings: list[str] = []
    video_path = resolve_path(raw["video"], root)
    if not video_path.exists():
        warnings.append(f"camera {cam_id}: video {raw['video']} not found")
    zones, zones_path = raw.get("zones") or None, None
    if zones:
        zones_path = resolve_path(zones, root)
        if not zones_path.exists():
            warnings.append(f"camera {cam_id}: zones file {zones} not found, running without zones")
            zones, zones_path = None, None
    for message in warnings:
        _warn(message, quiet)
    offset_s, sync_status = _sync_fields(raw, cam_id)

    return {
        "id": cam_id,
        "name": str(raw.get("name") or cam_id),
        "video": str(raw["video"]),
        "video_path": video_path,
        "start_time": start,
        "start_wall": format_wall(start),
        "preset": preset,
        "zones": str(zones) if zones else None,
        "zones_path": zones_path,
        "offset_s": offset_s,
        "sync_status": sync_status,
        "warnings": warnings,
    }


def _sync_fields(raw: dict, cam_id: str) -> tuple[float | None, str]:
    """Read and check the optional offset_s / sync_status of one registry entry."""
    offset = raw.get("offset_s")
    if offset in ("", None):
        offset = None
    else:
        try:
            offset = float(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"camera {cam_id}: offset_s {raw.get('offset_s')!r} is not a number") from exc
    status = str(raw.get("sync_status") or "").strip().lower()
    if not status:
        status = SYNC_USER_DECLARED if offset is not None else DEFAULT_SYNC_STATUS
    if status not in SYNC_STATUSES:
        raise ValueError(f"camera {cam_id}: sync_status {status!r} must be one of {', '.join(SYNC_STATUSES)}")
    if status != SYNC_UNAVAILABLE and offset is None:
        raise ValueError(f"camera {cam_id}: sync_status {status!r} needs an offset_s")
    return offset, status


def sync_info(camera: dict) -> dict:
    """{"offset_s", "sync_status"} of a camera dict (old dicts without the keys count as unavailable)."""
    status = str(camera.get("sync_status") or DEFAULT_SYNC_STATUS)
    offset = camera.get("offset_s")
    if status not in SYNC_STATUSES or (status != SYNC_UNAVAILABLE and offset is None):
        status = SYNC_UNAVAILABLE
    return {"offset_s": None if offset is None else float(offset), "sync_status": status}


def normalized_time(camera: dict, seconds: Any) -> tuple[float | None, str]:
    """Seconds into this camera's video -> (seconds on the shared site timeline, sync_status).

    Returns (None, "unavailable") when the camera has no declared alignment; the caller must then say
    TEMPORAL_UNAVAILABLE instead of comparing times across cameras.
    """
    info = sync_info(camera)
    if info["sync_status"] == SYNC_UNAVAILABLE or seconds is None:
        return None, SYNC_UNAVAILABLE
    return round(float(seconds) + info["offset_s"], 3), info["sync_status"]


def load_registry(path: str | Path, root: str | Path | None = None,
                  scenarios_dir: str | Path | None = None, quiet: bool = False) -> dict:
    """Read and validate a cameras.yaml. Returns {"site", "path", "cameras": [camera dicts]}.

    Raises ValueError for a broken registry (bad / duplicate id, no start_time, unknown preset).
    A missing video or zones file is only a warning.
    """
    global _LAST
    path = Path(path)
    if not path.exists():
        raise ValueError(f"camera registry {path} not found")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"camera registry {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list) or not data["cameras"]:
        raise ValueError(f"camera registry {path} needs a non-empty 'cameras:' list")
    root = Path(root) if root else ROOT
    scenarios_dir = Path(scenarios_dir) if scenarios_dir else SCENARIOS_DIR

    cameras, seen = [], set()
    for position, raw in enumerate(data["cameras"], start=1):
        camera = _camera(raw, position, root, scenarios_dir, quiet)
        if camera["id"] in seen:
            raise ValueError(f"camera id {camera['id']!r} is used twice")
        seen.add(camera["id"])
        cameras.append(camera)
    _LAST = cameras
    return {"site": str(data.get("site") or ""), "path": path, "cameras": cameras,
            "label": registry_label(path, data)}


def registry_label(path: str | Path, data: dict | None = None) -> str | None:
    """The dataset label of a registry: its `label:` key, else the `label` of a geometry.json next to it.

    Used to carry a warning such as "VIRTUAL MULTI-CAMERA DEMO - ..." into every output. None when neither exists.
    """
    if isinstance(data, dict) and str(data.get("label") or "").strip():
        return str(data["label"]).strip()
    geometry = Path(path).parent / "geometry.json"
    if geometry.is_file():
        try:
            label = json.loads(geometry.read_text(encoding="utf-8")).get("label")
        except (OSError, ValueError, AttributeError):
            return None
        return str(label).strip() if label else None
    return None


def load_cameras(path: str | Path, root: str | Path | None = None,
                 scenarios_dir: str | Path | None = None, quiet: bool = False) -> list[dict]:
    """Read and validate a cameras.yaml and return the list of camera dicts."""
    return load_registry(path, root=root, scenarios_dir=scenarios_dir, quiet=quiet)["cameras"]


def wall_clock(camera: dict, seconds: float) -> str:
    """Wall-clock ISO text of a moment `seconds` into this camera's video."""
    start = camera["start_time"] if isinstance(camera.get("start_time"), datetime) else parse_wall(camera["start_time"])
    return format_wall(start + timedelta(seconds=float(seconds)))


def video_seconds(camera: dict, wall: Any) -> float:
    """Inverse of wall_clock: seconds into this camera's video for a wall-clock time (may be negative)."""
    start = camera["start_time"] if isinstance(camera.get("start_time"), datetime) else parse_wall(camera["start_time"])
    return (parse_wall(wall) - start).total_seconds()


def camera_name(camera_id: str, cameras: list[dict] | None = None) -> str:
    """Human name of a camera (from `cameras`, else the registry loaded last); the id when unknown."""
    for camera in (cameras if cameras is not None else _LAST):
        if camera["id"] == camera_id:
            return camera["name"]
    return str(camera_id)


def get_camera(camera_id: str, cameras: list[dict] | None = None) -> dict | None:
    """The camera dict with this id (from `cameras`, else the registry loaded last), or None."""
    for camera in (cameras if cameras is not None else _LAST):
        if camera["id"] == camera_id:
            return camera
    return None


def main(argv=None) -> int:
    """Command-line check: print the cameras of a registry."""
    import argparse
    parser = argparse.ArgumentParser(description="Validate and list a camera registry (cameras.yaml)")
    parser.add_argument("registry")
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"site: {registry['site'] or '-'}  ({len(registry['cameras'])} cameras)")
    if registry.get("label"):
        print(f"label: {registry['label']}")
    for camera in registry["cameras"]:
        print(f"  {camera['id']:<14} {camera['name']:<20} start {camera['start_wall']}  preset {camera['preset']}"
              f"  video {camera['video']}  zones {camera['zones'] or '-'}"
              f"  sync {camera['sync_status']}"
              + ("" if camera["offset_s"] is None else f" (offset {camera['offset_s']:+.3f} s)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
