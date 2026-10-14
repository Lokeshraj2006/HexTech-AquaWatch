"""Shared helpers: config loading, time formatting, geometry, interval maths, JSON I/O.

Only depends on Python, NumPy and PyYAML (no OpenCV / PyTorch), so the core reasoning
modules that use it can be unit-tested without the vision stack.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent        # the project folder (this file is in aquawatch/)
CODE_DIR = Path(__file__).resolve().parent           # the code folder: modules, presets, tracker settings

BEHAVIORS = ("loitering", "zone_intrusion", "running", "fall", "crowding", "near_miss", "approaching", "aquatic_warning", "aquatic_distress",
             "submersion")
BEHAVIOR_LABELS = {"loitering": "LOITERING", "zone_intrusion": "ZONE INTRUSION", "running": "RUNNING",
                   "fall": "FALL", "crowding": "CROWDING", "near_miss": "NEAR MISS", "approaching": "APPROACHING", "aquatic_warning": "MAY NEED HELP",
                   "aquatic_distress": "AQUATIC DISTRESS", "submersion": "POSSIBLE SUBMERSION"}
# BGR for OpenCV drawing, hex for HTML / matplotlib.
BEHAVIOR_COLORS_BGR = {"loitering": (0, 165, 255), "zone_intrusion": (0, 0, 230), "running": (230, 0, 200),
                       "fall": (0, 69, 255), "crowding": (160, 160, 0), "near_miss": (60, 20, 220), "approaching": (0, 200, 255), "aquatic_warning": (11, 158, 245), "aquatic_distress": (0, 0, 208),
                       "submersion": (15, 4, 106)}
BEHAVIOR_COLORS_HEX = {"loitering": "#ffa500", "zone_intrusion": "#e60000", "running": "#c800e6",
                       "fall": "#ff4500", "crowding": "#00a0a0", "near_miss": "#dc143c", "approaching": "#ffc800", "aquatic_warning": "#f59e0b", "aquatic_distress": "#d00000",
                       "submersion": "#6a040f"}
DEFAULT_BEHAVIOR_NAMES = {"loitering": "Loitering", "zone_intrusion": "Zone intrusion", "running": "Running",
                          "fall": "Fall / person down", "crowding": "Crowding", "near_miss": "Near miss", "approaching": "Approaching you",
                          "aquatic_warning": "Swimmer may need help", "aquatic_distress": "Aquatic distress",
                          "submersion": "Possible submersion"}
# COCO class ids used by the presets (Ultralytics YOLO11 numbering).
CLASS_NAMES = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorbike", 5: "Bus", 7: "Truck", 14: "Bird",
               15: "Cat", 16: "Dog", 17: "Horse", 18: "Sheep", 19: "Cow"}
SCENARIOS_DIR = CODE_DIR / "scenarios"

DEFAULT_CONFIG = {
    "video": {"resize_width": 640, "frame_stride": 2, "max_seconds": 0},
    "model": {"weights": "yolo11n.pt", "device": "auto", "conf": 0.30, "iou": 0.50, "imgsz": 640,
              "tracker": "bytetrack_custom.yaml", "pose_weights": "yolo11n-pose.pt", "classes": [0]},
    "features": {"smoothing_s": 0.5, "speed_window_s": 0.5, "max_gap_s": 1.0,
                 "min_track_s": 1.0, "min_box_h_px": 20, "scale": "max_side"},
    "stitch": {"enabled": False, "max_gap_s": 5.0, "max_speed_bh_s": 1.0, "slack_bh": 1.0, "max_size_ratio": 3.0,
               "max_jump_bh": 3.0},
    "behaviors": {
        "loitering": {"enabled": True, "radius_bh": 0.5, "min_duration_s": 10.0, "bridge_gap_s": 3.0, "active_hours": ""},
        "zone_intrusion": {"enabled": True, "min_duration_s": 1.0, "active_hours": ""},
        "running": {"enabled": True, "start_speed_bh_s": 1.8, "end_speed_bh_s": 1.4, "min_duration_s": 1.0, "active_hours": ""},
        "baseline": {"enabled": True, "min_tracks": 4, "z_threshold": 3.0},
        "fall": {"enabled": True, "down_min_aspect": 1.2, "upright_max_aspect": 0.8,
                 "transition_s": 2.0, "min_down_s": 2.0, "active_hours": ""},
        "crowding": {"enabled": True, "min_count": 5, "min_duration_s": 5.0, "whole_frame": False, "active_hours": ""},
        "near_miss": {"enabled": True, "vulnerable_classes": [0], "other_classes": [0, 1, 2, 3, 5, 7],
                      "near_distance_bh": 0.6, "min_rel_speed_bh_s": 0.8, "min_closing_speed_bh_s": 0.3,
                      "ttc_s": 1.0, "contact_bh": 0.15, "max_scale_diff": 0.35, "min_duration_s": 0.0,
                      "active_hours": ""},
        "approaching": {"enabled": False, "classes": [0, 1, 2, 3, 16], "window_s": 0.5, "min_growth_per_s": 0.05,
                        "ttc_s": 2.5, "center_band": 0.35, "min_size_frac": 0.25, "min_duration_s": 0.3,
                        "uniform_growth_ratio": 2.0, "cooldown_s": 3.0, "active_hours": ""},
        "aquatic_warning": {"enabled": False, "active_hours": ""},
        "aquatic_distress": {"enabled": False, "active_hours": ""},
        "submersion": {"enabled": False, "active_hours": ""},
    },
    "events": {"merge_gap_s": 1.5, "near_miss_ratio": 0.7,
               "confidence_weights": {"margin": 0.4, "duration": 0.3, "detection": 0.3}},
    "pose": {"max_frames_per_event": 12, "min_keypoint_conf": 0.4, "min_pose_frames": 4,
             "run_leg_spread_bh": 0.30, "run_lean_deg": 6.0, "upright_max_lean_deg": 25.0},
    "output": {"annotated_video": True, "trail_s": 3.0, "heatmap": True, "timeline": True},
    "chains": {"enabled": True, "max_gap_s": 15.0},
    "highlights": {"enabled": True, "pad_s": 1.5, "max_segment_s": 8.0, "title_s": 2.0},
    "feed_health": {"enabled": True, "sample_hz": 2.0, "problem_s": 2.0, "dark_mean": 20, "blank_std": 4,
                    "frozen_change": 0.25, "end_missing_s": 1.0, "gap_s": 1.0, "view_warnings": False, "move_frac": 0.02,
                    "top_down_ratio": 1.3, "view_min_boxes": 20},
    "camera": {"name": ""},
    "privacy": {"enabled": False, "mode": "head", "head_fraction": 0.28, "style": "pixelate"},
    "time_of_day": {"start_time": ""},
    "pool": {"enabled": False, "state_classes": {}, "sample_s": 0.5, "window_s": 2.0,
             "horizontal_aspect": 1.3, "head_area_frac": 0.45, "movement_low_bh_s": 0.25,
             "movement_high_bh_s": 0.8, "displacement_low_bh_s": 0.15, "pose": True, "pose_hz": 6.0,
             "kp_conf": 0.35, "arm_min_hz": 0.6, "arm_min_amp": 0.08, "arm_calm_amp": 0.04,
             "head_unstable": 0.06, "shrink_ratio": 0.55, "edge_zone_words": ["edge", "wall"],
             "water_zone_words": ["water", "pool"], "default_location": "Pool",
             "weights": {"vertical": 0.25, "low_displacement": 0.25, "repeated_arms": 0.25,
                         "unstable_head": 0.15, "transition": 0.10, "stationary": 0.15},
             "watch_risk": 0.3, "warning_risk": 0.55, "alert_risk": 0.7, "release_risk": 0.5,
             "hold_s": 5.0, "lookback_s": 30.0, "resting_factor": 0.3, "treading_cap": 0.45, "stationary_lifts_treading_cap": False,
             "submersion_lost_s": 1.5, "submersion_head_s": 1.0,
             "early_warning_window_s": 5.0, "early_warning_fraction": 0.5, "early_warning_cooldown_s": 60.0,
             "stationary_s": 4.0, "standing_zone_words": ["shallow", "steps", "kids"], "passive_s": 8.0,
             "vanish_enabled": False, "vanish_min_seen_s": 3.0, "vanish_lost_s": 3.0, "vanish_edge_bh": 1.0,
             "vanish_reappear_bh": 2.0, "head_relative_to_neighbours": False, "head_neighbour_bh": 4.0,
             "head_min_neighbours": 2, "same_spot_enabled": False, "same_spot_min_s": 5.0,
             "same_spot_max_s": 20.0, "same_spot_bh": 1.5, "mark_spot_s": 4.0,
             "vertical_above_deg": 60.0, "horizontal_below_deg": 30.0, "posture_hold_s": 1.0,
             "arm_peak_prominence": 0.08, "arm_none_hz": 0.3, "head_submerged_below": 0.2,
             "head_up_min": 0.08, "ema_alpha": 0.7, "kp_max_gap": 5, "jump_factor": 0.8, "min_box_px": 40},
    "announce": {"enabled": False, "min_risk": 0.7, "behaviors": ["aquatic_warning", "aquatic_distress", "submersion"],
                 "warning_min_risk": 0.55, "repeat": 2, "voice": "", "rate": 0,
                 "message": "Attention lifeguard. {who} may be in distress. {where}.",
                 "submersion_message": "Urgent. {who} may have gone under. {where}. Check now.",
                 "under_water_message": "Urgent. {who} may have been under the water. {where}. Go and check now.",
                 "warning_message": "Lifeguard, please check {who}. {where}. They may need help."},
    "scenario": {"name": "campus", "title": "Campus corridor (restricted zone)", "entity_word": "Person",
                 "unit_word": "body-heights", "behavior_names": {}},
}


# --------------------------------------------------------------------------- config

def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of `base` with `override` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, scenario: str | None = None) -> dict:
    """Load config.yaml (default: next to this file) merged over DEFAULT_CONFIG.

    If `scenario` is given (a preset name like "livestock" or a path to a YAML file),
    that preset is deep-merged on top, so it can change any threshold or wording.
    """
    path = Path(path) if path else ROOT / "config.yaml"
    user = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
    cfg = deep_merge(DEFAULT_CONFIG, user)
    if scenario:
        preset = Path(scenario)
        if not preset.suffix:
            preset = SCENARIOS_DIR / f"{scenario}.yaml"
        if not preset.exists():
            names = sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml")) if SCENARIOS_DIR.exists() else []
            raise FileNotFoundError(f"Scenario '{scenario}' not found. Available: {', '.join(names) or 'none'}")
        with open(preset, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return cfg


def list_scenarios() -> list[str]:
    """Names of the presets in scenarios/."""
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml")) if SCENARIOS_DIR.exists() else []


def entity_word(cfg: dict | None) -> str:
    """What the tracked things are called in this scenario ("Person", "Animal", "Vehicle")."""
    return ((cfg or {}).get("scenario") or {}).get("entity_word") or "Person"


def entity_name(cfg: dict | None, track_id, cls: int | None = None) -> str:
    """Display name for a tracked entity, e.g. "Person #3" or "Car #7". None -> "Group".

    When the scenario tracks several classes (e.g. traffic), the class name is used;
    otherwise the scenario's entity word ("Person", "Animal", ...).
    """
    if track_id is None:
        return "Group"
    classes = ((cfg or {}).get("model") or {}).get("classes") or [0]
    # a state detector's classes are body states of one entity (Swimmer), not COCO kinds of object
    if cls is not None and len(classes) > 1 and not state_classes(cfg):
        return f"{CLASS_NAMES.get(int(cls), 'Object')} #{track_id}"
    return f"{entity_word(cfg)} #{track_id}"


def tracker_yaml_sha1(name) -> str:
    """SHA-1 of a ByteTrack yaml (looked up like tracker.py: absolute, code folder, project root, cwd); '' when not found."""
    import hashlib
    p = Path(str(name))
    for candidate in ([p] if p.is_absolute() else [CODE_DIR / p, ROOT / p, Path.cwd() / p]):
        if candidate.is_file():
            return hashlib.sha1(candidate.read_bytes()).hexdigest()
    return ""


def tracking_settings(cfg: dict) -> dict:
    """The detector / tracker settings a detections.json records (tracker.py, test fixtures) and --reuse checks."""
    m = cfg["model"]
    return {"conf": float(m["conf"]), "iou": float(m["iou"]), "imgsz": int(m["imgsz"]), "tracker": str(m["tracker"]),
            "tracker_sha1": tracker_yaml_sha1(m["tracker"])}


def tracking_settings_mismatch(header: dict, cfg: dict) -> str:
    """Why cached detections were made with other detector / tracker settings ('' when they match): conf, iou,
    imgsz and the ByteTrack yaml (its content). A cache from before these were recorded (2026-10-12) is not reused
    (bug hunt 2026-10-11, tracking medium; PREREGISTRATION_bug_group_C.md step C6)."""
    m = cfg["model"]
    if any(k not in header for k in ("conf", "iou", "imgsz", "tracker_sha1")):
        return "cached detections do not record their detector / tracker settings (made before 2026-10-12)"
    for key in ("conf", "iou"):
        if abs(float(header[key]) - float(m[key])) > 1e-9:
            return f"detector {key} changed ({header[key]} -> {m[key]})"
    if int(header["imgsz"]) != int(m["imgsz"]):
        return f"detector image size changed ({header['imgsz']} -> {m['imgsz']})"
    if header["tracker_sha1"] != tracker_yaml_sha1(m["tracker"]):
        return f"tracker settings changed ({m['tracker']})"
    return ""


def state_classes(cfg: dict | None) -> dict[int, str]:
    """{class id: body state} of a custom state detector (pool.state_classes); {} for the stock COCO detector."""
    raw = ((cfg or {}).get("pool") or {}).get("state_classes") or {}
    return {int(k): str(v) for k, v in raw.items()}


def class_labels(cfg: dict | None) -> list[str]:
    """Plain names of the tracked classes, for captions: COCO names ("person", "car"), or the body states of a
    custom state detector ("swimmer horizontal", "head only")."""
    classes = ((cfg or {}).get("model") or {}).get("classes") or [0]
    states = state_classes(cfg)
    if states:
        return [states.get(int(c), str(c)).replace("_", " ") for c in classes]
    return [CLASS_NAMES.get(int(c), str(c)).lower() for c in classes]


def weights_file(name) -> Path | None:
    """The detector's weights on disk (absolute path, or relative to the project folder); None when absent
    (e.g. yolo11n.pt before Ultralytics has downloaded it). An exported OpenVINO model is a FOLDER
    (<name>_openvino_model, made by quantize.py): it counts when it holds its .xml model file."""
    p = Path(str(name))
    p = p if p.is_absolute() else ROOT / p
    if p.is_dir():
        return p if any(p.glob("*.xml")) else None
    return p if p.is_file() else None


def fp32_export_source(weights) -> tuple[str, str] | None:
    """(source weights name, source SHA-1) when `weights` is an FP32 export made by quantize.py (its manifest.json sits
    next to the exported model), else None. An FP32 export computes the same boxes as its source (checked: F1 1.000,
    outputs/quant/quant_eval.json), so detections cached with the source stay valid. INT8 exports never qualify."""
    found = weights_file(weights)
    if found is None or not (found.parent / "manifest.json").is_file():
        return None
    manifest = read_json(found.parent / "manifest.json") or {}
    if manifest.get("precision") != "fp32" or not manifest.get("source"):
        return None
    return str(manifest["source"]), str(manifest.get("source_sha1") or "")


def same_detector(cached_model, cached_sha1, weights, weights_sha1_now: str = "") -> bool:
    """May detections made with `cached_model` (and its SHA-1, when recorded) be reused with `weights` now? Same file
    name and same content, or `weights` is an FP32 export of exactly that file."""
    if not cached_model:
        return True                                        # an old file without the detector name: not checked
    export = fp32_export_source(weights)
    if export and Path(str(cached_model)).name == Path(export[0]).name:
        return not cached_sha1 or not export[1] or cached_sha1 == export[1]
    if Path(str(cached_model)).name != Path(str(weights)).name:
        return False
    return not (cached_sha1 and weights_sha1_now and cached_sha1 != weights_sha1_now)


def weights_sha1(name) -> str:
    """SHA-1 of the detector's weights file, "" when it is not on disk. Stored in detections.json so that --reuse
    can tell a retrained best.pt from the old one (both have the same file name)."""
    p = weights_file(name)
    if p is None:
        return ""
    files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".xml", ".bin")) if p.is_dir() else [p]
    h = hashlib.sha1()
    try:
        for f in files:                    # an OpenVINO folder: its .xml (graph) and .bin (weights) together
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
    except OSError:                        # locked (e.g. being copied in) or unreadable: "" = skip the check
        return ""
    return h.hexdigest()


def detections_fingerprint(rows) -> str:
    """Short hash of which track ID was seen on which frame (detection rows: frame, t, track id, ...). Cached pose
    keypoints belong to one set of tracks; this key stops them being attached to another set (new IDs after a
    re-run, another --max-seconds, a retrained model)."""
    h = hashlib.sha1()
    for row in rows or []:
        h.update(f"{int(row[0])}:{int(row[2])};".encode())
    return h.hexdigest()[:16]


def behavior_name(cfg: dict | None, key: str) -> str:
    """Display name of a behaviour in this scenario (e.g. running -> "Speeding" for traffic)."""
    names = ((cfg or {}).get("scenario") or {}).get("behavior_names") or {}
    return names.get(key) or DEFAULT_BEHAVIOR_NAMES.get(key) or key.replace("_", " ").title()


def unit_name(cfg: dict | None) -> str:
    """Name of the size unit speeds/distances are measured in ("body-heights", "body-lengths")."""
    return ((cfg or {}).get("scenario") or {}).get("unit_word") or "body-heights"


# --------------------------------------------------------------------------- time

def fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss (or h:mm:ss for long videos). Rounds down."""
    if seconds is None or not math.isfinite(seconds):
        return "--:--"
    s = int(max(0.0, seconds) + 0.02)   # +0.02: 14.998 s (frame rounding) shows as 00:15, not 00:14
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def fmt_time_precise(seconds: float) -> str:
    """Format seconds as mm:ss.s, for overlays and evidence text."""
    if seconds is None or not math.isfinite(seconds):
        return "--:--.-"
    seconds = max(0.0, seconds)
    m, sec = divmod(seconds, 60)
    return f"{int(m):02d}:{sec:04.1f}"


def parse_clock(text) -> float | None:
    """'22:05' or '22:05:30' -> seconds after midnight; '' / None -> None. Raises ValueError if malformed."""
    if text is None or str(text).strip() == "":
        return None
    parts = str(text).strip().split(":")
    if len(parts) not in (2, 3) or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"time must look like HH:MM or HH:MM:SS, got {text!r}")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        raise ValueError(f"time out of range: {text!r}")
    return float(h * 3600 + m * 60 + s)


def parse_hours(spec) -> list[tuple[float, float]]:
    """'22:00-06:00' or '08:00-12:00, 14:00-18:00' -> [(start, end)] in seconds after midnight.

    A window may wrap past midnight (22:00-06:00). '' means "always" and gives [].
    """
    if spec is None or str(spec).strip() == "":
        return []
    windows = []
    for part in str(spec).split(","):
        if "-" not in part:
            raise ValueError(f"active hours must look like 22:00-06:00, got {part.strip()!r}")
        a, b = part.split("-", 1)
        windows.append((parse_clock(a), parse_clock(b)))
    return windows


def fmt_clock(seconds_after_midnight: float) -> str:
    """Seconds after midnight -> 'HH:MM:SS' (wraps past 24 h)."""
    s = int(seconds_after_midnight) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def in_hours(clock_s: float, windows) -> bool:
    """Is this clock time (seconds after midnight) inside any window? No windows = always."""
    if not windows:
        return True
    c = clock_s % 86400
    for a, b in windows:
        if (a <= c < b) if a <= b else (c >= a or c < b):       # second form: window wraps midnight
            return True
    return False


def span_in_hours(start_clock_s: float, t0: float, t1: float, windows) -> bool:
    """Does the video span [t0, t1] (seconds into the video) touch any active window? Checked every second."""
    if not windows:
        return True
    t = t0
    while t <= t1:
        if in_hours(start_clock_s + t, windows):
            return True
        t += 1.0
    return in_hours(start_clock_s + t1, windows)


# --------------------------------------------------------------------------- geometry

def point_in_polygon(x: float, y: float, poly) -> bool:
    """Ray-casting test: is (x, y) inside the polygon [(x, y), ...]?"""
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def distance_to_polygon_edge(x: float, y: float, poly) -> float:
    """Shortest distance (pixels) from (x, y) to any edge of the polygon."""
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        u = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg2))
        px, py = x1 + u * dx, y1 + u * dy
        best = min(best, math.hypot(x - px, y - py))
    return best


def load_zones(path: str | Path | None, proc_size) -> list[dict]:
    """Load zones.json and scale its points to the processing frame size.

    Accepted formats:
      {"image_size": [W, H], "zones": [{"name": "restricted", "points": [[x, y], ...]}]}
      {"zones": [...]}            (points already in processing pixels)
      [[x, y], ...]               (a single unnamed zone in processing pixels)
    Returns [{"name": str, "points": [(x, y), ...]}]; [] when no path / file.
    """
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        return []
    data = read_json(path)
    if isinstance(data, list):
        data = {"zones": [{"name": "zone", "points": data}]}
    if not isinstance(data, dict):
        raise ValueError("expected an object with a 'zones' list (see zones.example.json)")
    zones = data.get("zones", [])
    sx = sy = 1.0
    if data.get("image_size") and proc_size:
        w, h = (float(v) for v in data["image_size"])
        if w <= 0 or h <= 0:
            raise ValueError(f"image_size must be two positive numbers, got {data['image_size']}")
        sx, sy = proc_size[0] / w, proc_size[1] / h
    out = []
    for i, z in enumerate(zones):
        pts = []
        for p in z.get("points", []):
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"zone {i + 1}: every point must be [x, y], got {p!r}")
            pts.append((float(p[0]) * sx, float(p[1]) * sy))
        if len(pts) >= 3:
            out.append({"name": z.get("name") or f"zone{i + 1}", "points": pts})
    return out


def save_zones(path: str | Path, zones: list[dict], image_size) -> None:
    """Write zones (points in original video pixels) in the standard zones.json format."""
    write_json(path, {"image_size": [int(image_size[0]), int(image_size[1])],
                      "zones": [{"name": z["name"], "points": [[round(x, 1), round(y, 1)] for x, y in z["points"]]}
                                for z in zones]})


# --------------------------------------------------------------------------- signals / intervals

def moving_average(values: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    """Centred moving average over a time window (NaN-safe). Works on 1-D or (n, k) arrays."""
    values = np.asarray(values, dtype=float)
    t = np.asarray(t, dtype=float)
    if len(t) == 0 or window_s <= 0:
        return values.copy()
    half = window_s / 2.0
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    out = np.empty_like(values)
    for i in range(len(t)):
        chunk = values[lo[i]:hi[i]]
        out[i] = np.nanmean(chunk, axis=0) if len(chunk) else values[i]
    return out


def mask_to_intervals(t, mask, max_gap_s: float | None = None) -> list[tuple[float, float, int, int]]:
    """Turn a boolean mask over samples into runs: [(start_s, end_s, i0, i1)] (i1 inclusive).

    If max_gap_s is given, a time gap between consecutive samples larger than it also
    ends a run (so a person who vanished is not counted as present).
    """
    t = np.asarray(t, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    runs = []
    i0 = None
    for i in range(len(t)):
        gap = max_gap_s is not None and i > 0 and (t[i] - t[i - 1]) > max_gap_s
        if i0 is not None and (not mask[i] or gap):
            runs.append((float(t[i0]), float(t[i - 1]), i0, i - 1))
            i0 = None
        if mask[i] and i0 is None:
            i0 = i
    if i0 is not None:
        runs.append((float(t[i0]), float(t[len(t) - 1]), i0, len(t) - 1))
    return runs


def merge_intervals(intervals, gap_s: float) -> list[tuple[float, float]]:
    """Merge (start, end) intervals whose gap is <= gap_s. Returns sorted merged list."""
    items = sorted((float(a), float(b)) for a, b in intervals)
    merged: list[list[float]] = []
    for a, b in items:
        if merged and a - merged[-1][1] <= gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def robust_z(values) -> tuple[np.ndarray, float, float]:
    """Robust z-scores using median and MAD. Returns (z, median, mad)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v, float("nan"), float("nan")
    med = float(np.nanmedian(v))
    mad = float(np.nanmedian(np.abs(v - med)))
    scale = 1.4826 * mad if mad > 1e-9 else (np.nanstd(v) or 1e-9)
    return (v - med) / scale, med, mad


def clip01(x: float) -> float:
    """Clamp to [0, 1]; NaN becomes 0."""
    if x is None or not math.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


# --------------------------------------------------------------------------- misc

def id_color(track_id: int) -> tuple[int, int, int]:
    """Stable, distinct BGR colour for a track ID."""
    golden = 0.618033988749895
    hue = (int(track_id) * golden) % 1.0
    # simple HSV->RGB with s=0.65, v=0.95
    i = int(hue * 6)
    f = hue * 6 - i
    v, s = 0.95, 0.65
    p, q, tt = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, tt, p), (q, v, p), (p, v, tt), (p, q, v), (tt, p, v), (v, p, q)][i % 6]
    return int(b * 255), int(g * 255), int(r * 255)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _clean_nans(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nans(v) for v in obj]
    return obj


def write_json(path: str | Path, obj) -> None:
    """Write JSON (NumPy-safe; NaN/inf become null)."""
    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_clean_nans(json.loads(json.dumps(obj, cls=_NumpyEncoder))), fh, indent=2, ensure_ascii=False)


def read_json(path: str | Path):
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
