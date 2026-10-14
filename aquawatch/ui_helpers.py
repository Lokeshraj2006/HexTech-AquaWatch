"""Helpers for the Streamlit front end (app.py): find files, build the run.py command, stream its output,
read a results folder.

Design rules (see CLAUDE.md):
  * No Streamlit import here, so every function can be unit-tested without a browser.
  * No torch / ultralytics here. The heavy work (YOLO, tracking, rendering) runs in a `run.py` subprocess,
    so the web page stays fast and the core modules are wrapped, never changed.
  * OpenCV is imported lazily and only to read the first frame of a video for the preview.
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile
from collections import deque
from pathlib import Path

import utils

ROOT = utils.ROOT
SAMPLES_DIR = ROOT / "samples"
UPLOAD_DIR = SAMPLES_DIR / "uploads"
OUTPUTS_DIR = ROOT / "outputs"
UI_DIR = OUTPUTS_DIR / "_ui"                       # zones written by the UI (rectangle, uploaded zones.json)
CACHE_DIR = UI_DIR / "cache"                       # old detections.json files kept before a run replaces them
PREVIEW_DIR = UI_DIR / "preview"                   # browser-playable (H.264) copies of input videos for the preview
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov", ".webm"}   # containers a browser can open (not .avi / .mkv)
LAST_RUN_FILE = UI_DIR / "last_run.json"           # the latest run, so a browser reload still shows its results
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
MAX_COORD = 1_000_000                              # a zone point further out than this is certainly a mistake
MAX_LOG_LINES = 5000                               # lines of run.py output kept in memory
MAX_LINE_CHARS = 2000                              # a single output line is cut after this many characters
ZONE_DIRS = ("samples", "tests", "data/examples")       # where *zones*.json files are looked for
RESULT_ROOTS = ("outputs", "data/examples")             # where result folders (with an events.json) are looked for

SEVERITY_COLORS = {"high": "red", "medium": "orange", "low": "blue"}
ZONE_BGR = (0, 0, 230)                             # zone colour for the preview (BGR, same red as the report)


# --------------------------------------------------------------------------- paths and files

def rel_path(path) -> str:
    """Path relative to the project root with forward slashes (the absolute path when it is outside the project)."""
    full = os.path.abspath(str(path))
    try:
        return Path(full).relative_to(ROOT).as_posix()
    except ValueError:
        return full


def mtime(path) -> float:
    """Modification time of a file, 0.0 when it does not exist (used as part of cache keys)."""
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def safe_stem(name: str, default: str = "video") -> str:
    """File name without folders or extension, with spaces and odd characters replaced by "_".

    A name made only of odd characters gets a short hash, so two different uploads do not overwrite each other.
    """
    base = Path(str(name).replace("\\", "/")).name            # drop any folder part
    stem = os.path.splitext(base)[0]
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:80]
    if not clean:
        clean = f"{default}_{hashlib.sha1(str(name).encode('utf-8', 'replace')).hexdigest()[:6]}"
    if clean.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(COM|LPT)\d", clean.upper()):
        clean = "_" + clean                                   # reserved device names on Windows
    return clean


def safe_filename(name: str, default: str = "video") -> str:
    """Make an uploaded file name safe to store: no folders, no spaces or odd characters, a video extension.

    "My clip (final)#2.MP4" -> "My_clip_final_2.mp4"; "../../etc/passwd" -> "passwd.mp4".
    """
    ext = os.path.splitext(Path(str(name).replace("\\", "/")).name)[1].lower()
    return safe_stem(name, default) + (ext if ext in VIDEO_EXTS else ".mp4")


def _file_sha1(path: Path) -> str:
    """SHA-1 of a file, read in 1 MB pieces (uploads can be hundreds of MB)."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for piece in iter(lambda: fh.read(1 << 20), b""):
            digest.update(piece)
    return digest.hexdigest()


def save_upload(name: str, data) -> Path:
    """Write an uploaded video (bytes or a memoryview) to samples/uploads/<safe name> and return the path.

    The same bytes under the same name are not written again. A DIFFERENT video with a name that is already taken
    gets a short content tag ("clip_3f9a1c.mp4"), because cached detections are matched by file name: reusing
    the old name would make --reuse load the detections of the other video.
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / safe_filename(name)
    digest = hashlib.sha1(data).hexdigest()
    if dest.exists() and _file_sha1(dest) != digest:
        dest = dest.with_name(f"{dest.stem}_{digest[:6]}{dest.suffix}")
    if not (dest.exists() and _file_sha1(dest) == digest):
        with open(dest, "wb") as fh:
            fh.write(data)
    return dest


def discard_upload(path) -> bool:
    """Delete a file from samples/uploads/ (an upload that turned out not to be a video). True when it was removed."""
    path = Path(path)
    try:
        path.resolve().relative_to(UPLOAD_DIR.resolve())          # never delete anything outside the uploads folder
        path.unlink()
        return True
    except (ValueError, OSError):
        return False


def list_videos() -> list[tuple[str, Path]]:
    """(label, path) for every video in samples/ and, labelled "uploads/...", in samples/uploads/."""
    found = []
    for folder, prefix in ((SAMPLES_DIR, ""), (SAMPLES_DIR / "pexels", "pexels/"), (UPLOAD_DIR, "uploads/")):
        if folder.is_dir():
            for p in sorted(folder.iterdir(), key=lambda q: q.name.lower()):
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    found.append((prefix + p.name, p))
    return found


def list_zone_files() -> list[Path]:
    """Every *zones*.json under samples/, tests/ and examples/ (sorted by relative path)."""
    found = []
    for name in ZONE_DIRS:
        base = ROOT / name
        if base.is_dir():
            found += [p for p in base.rglob("*zones*.json") if p.is_file()]
    return sorted(found, key=lambda p: rel_path(p).lower())


def suggest_zone_file(video: Path | None, zone_files: list[Path]) -> Path | None:
    """The zones file named after the video: "<stem>_zones.json", or "one-by-one_zones.json" for "one-by-one-person-...mp4"."""
    if video is None:
        return None
    stem = video.stem.lower()
    best = None
    for z in zone_files:
        prefix = re.sub(r"[_-]?zones?(\.example)?$", "", z.stem.lower())
        if not prefix or prefix == "zones":
            continue
        if stem == prefix or stem.startswith(prefix + "-"):       # exact name, or a "-..." continuation of it
            if best is None or len(prefix) > best[0]:
                best = (len(prefix), z)
    return best[1] if best else None


def list_result_dirs() -> list[Path]:
    """Folders directly inside outputs/ and examples/ that hold an events.json (finished results)."""
    found = []
    for name in RESULT_ROOTS:
        base = ROOT / name
        if base.is_dir():
            for d in sorted(base.iterdir(), key=lambda q: q.name.lower()):
                if d.is_dir() and not d.name.startswith("_") and (d / "events.json").is_file():
                    found.append(d)
    return found


def safe_child(folder, relative) -> Path | None:
    """folder / relative if that file exists and stays inside the folder (events.json is data, not trusted)."""
    if not relative:
        return None
    root = Path(folder).resolve()
    try:
        path = (root / str(relative)).resolve()
        path.relative_to(root)
    except (ValueError, OSError):
        return None
    return path if path.is_file() else None


def zip_folder_bytes(folder) -> bytes:
    """The whole results folder as one zip (built in memory). Already-compressed media is stored, not deflated."""
    folder = Path(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            kind = zipfile.ZIP_STORED if f.suffix.lower() in (".mp4", ".jpg", ".png") else zipfile.ZIP_DEFLATED
            try:
                zf.write(f, f"{folder.name}/{f.relative_to(folder).as_posix()}", compress_type=kind)
            except OSError:
                continue                                    # a file that vanished or is locked: skip it
    return buf.getvalue()


# --------------------------------------------------------------------------- scenarios

def scenario_summary(name: str) -> dict:
    """Title, enabled rules and tracked classes of a preset, for the sidebar. Never raises."""
    try:
        cfg = utils.load_config(None, name)
    except Exception as exc:                                  # a broken preset must not break the page
        return {"name": name, "title": f"{name} (cannot be loaded)", "rules": [], "classes": [], "error": str(exc)}
    rules = [utils.behavior_name(cfg, key) for key in utils.BEHAVIORS if cfg["behaviors"][key].get("enabled", True)]
    classes = utils.class_labels(cfg)
    return {"name": name, "title": cfg["scenario"]["title"], "rules": rules, "classes": classes}


SIDEBAR_PRESETS = ("pool_trained", "pool", "pool_hd")   # AquaWatch: the sidebar offers only the pool presets (the engine keeps all)


def sidebar_presets(names: list[str]) -> list[str]:
    """The presets the AquaWatch sidebar offers: the pool ones (all presets if none of them exists)."""
    return [n for n in names if n in SIDEBAR_PRESETS] or list(names)


def sidebar_result_dirs(dirs: list) -> list:
    """Results folders for "Open existing results": pool runs only (they hold pool_dashboard.json)."""
    return [d for d in dirs if (Path(d) / "pool_dashboard.json").is_file()] or list(dirs)


def trained_model_ready(names: list[str]) -> bool:
    """Is the pool_trained preset there AND its weights file (training/runs/swimmer/weights/best.pt) on disk?"""
    return "pool_trained" in names and weights_present(scenario_needs("pool_trained").get("weights_path", ""))


def default_preset(names: list[str]) -> str | None:
    """The preset the page starts with: our trained swimmer model when its weights are here (real footage is the
    main use), else the stock pool preset (a fresh copy without best.pt still works), else the first preset."""
    if trained_model_ready(names):
        return "pool_trained"
    if "pool" in names:
        return "pool"
    return names[0] if names else None


def restored_preset(saved: str | None, video, names: list[str]) -> str | None:
    """The preset to put back in the sidebar after a reload. Normally the one the last run used (a reload is not a
    different job), but a last run with the stock pool detector on REAL footage reopens with the trained swimmer
    model when it is available: it is the default for real footage, and the stock one splits swimmers into many IDs."""
    if saved == "pool" and video and not _is_synthetic(video) and trained_model_ready(names):
        return "pool_trained"
    if saved == "pool_trained" and video and _is_synthetic(video) and "pool" in names:
        return "pool"                            # drawn test clips: only the stock detector sees their swimmers
    return saved


def is_synthetic(video) -> bool:
    """Is this one of the drawn test clips (samples/synthetic*.mp4)?"""
    return _is_synthetic(video)


def _is_synthetic(video) -> bool:
    """The drawn test clips (samples/synthetic*.mp4) are made for the stock detector."""
    return Path(str(video)).name.lower().startswith("synthetic")


def preset_hint(scenario: str | None, video, names: list[str]) -> str:
    """A tip when the chosen pool preset does not suit the chosen video ("" otherwise).

    - a drawn synthetic clip with the trained model: it finds nothing in drawn swimmers, use the stock preset;
    - real footage with the stock COCO detector while the trained model is available: the stock detector is not a
      swimmer detector (it misses swimmers and adds duplicate boxes), use the trained one."""
    if video is None:
        return ""
    if scenario == "pool_trained" and _is_synthetic(video) and "pool" in names:
        return ("This is a drawn test clip: the trained swimmer model finds nothing in drawn swimmers. Pick **"
                + scenario_summary("pool")["title"] + "** for it.")
    if scenario == "pool" and not _is_synthetic(video) and trained_model_ready(names):
        return ("For real pool footage pick **" + scenario_summary("pool_trained")["title"] + "**: the stock detector "
                "is not a swimmer detector, so it can miss swimmers or count one swimmer twice.")
    return ""


def default_stride(scenario: str | None = None) -> int:
    """frame_stride of config.yaml with the preset on top (navigation uses 1, the others 2).

    The sidebar shows it as the default; the --stride flag is only sent when the user changes it.
    """
    try:
        return int(utils.load_config(None, scenario)["video"]["frame_stride"])
    except Exception:
        return 2


def preset_classes(name: str) -> list[int]:
    """COCO class ids a preset tracks (needed to find compatible cached detections)."""
    try:
        return [int(c) for c in utils.load_config(None, name)["model"]["classes"]]
    except Exception:
        return [0]


def scenario_needs(name: str | None) -> dict:
    """What cached detections must match for this preset (the same fields run.py checks before --reuse).

    {"classes": [0], "stride": 2, "resize_width": 640, "weights": "yolo11n.pt", "max_seconds": 0.0,
     "weights_path": "yolo11n.pt", "weights_sha1": "..."}
    `weights` is the file name (what run.py compares); `weights_path` is the configured path (for the "is it here?"
    check) and `weights_sha1` its content hash ("" when not on disk), so a retrained best.pt is not mistaken for the old one.
    """
    try:
        cfg = utils.load_config(None, name)
        return {"classes": [int(c) for c in cfg["model"]["classes"]], "stride": int(cfg["video"]["frame_stride"]),
                "resize_width": int(cfg["video"]["resize_width"] or 0), "weights": Path(str(cfg["model"]["weights"])).name,
                "max_seconds": float(cfg["video"].get("max_seconds") or 0.0),
                "weights_path": str(cfg["model"]["weights"]), "weights_sha1": utils.weights_sha1(cfg["model"]["weights"])}
    except Exception:                                          # a broken preset: fall back to the base defaults
        return {"classes": [0], "stride": 2, "resize_width": 640, "weights": "yolo11n.pt", "max_seconds": 0.0,
                "weights_path": "yolo11n.pt", "weights_sha1": ""}


# --------------------------------------------------------------------------- video preview (OpenCV, lazy)

def video_facts(path) -> dict | None:
    """Width, height, fps, frame count and duration of a video; None when OpenCV cannot open it."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, frame = cap.read()
        if ok and frame is not None:                 # trust the real picture over the header (rotated phone videos)
            height, width = frame.shape[:2]
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        return None
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    return {"width": width, "height": height, "fps": fps, "frames": frames, "duration_s": duration}


def first_frame(path, max_width: int = 960):
    """First frame as a BGR array shrunk to at most `max_width` px wide. Returns (frame, scale) or (None, 1.0).

    `scale` = shown width / original width, so zone points in original pixels can be drawn on the picture.
    """
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None, 1.0
    width = frame.shape[1]
    scale = min(1.0, max_width / float(width))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(round(width * scale)), int(round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
    return frame, scale


def browser_video(path) -> Path | None:
    """A file of this video that a browser can play: the video itself when it is H.264 / VP8 / VP9 / AV1 (phone
    clips), else an H.264 copy made once and kept in outputs/_ui/preview/ (OpenCV's mp4v files, .avi, ...).
    None when it cannot be converted; the page then shows the first frame instead. Never raises."""
    try:
        from render import BROWSER_CODECS, transcode_h264, video_codec   # lazy: OpenCV, only for the preview
        path = Path(path)
        # browsers play H.264 / VP8 / VP9 / AV1 only inside these containers (not .avi / .mkv, common for CCTV)
        if path.suffix.lower() in BROWSER_CONTAINERS and video_codec(path) in BROWSER_CODECS:
            return path
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        # one name per SOURCE file (samples/clip.mp4 and samples/uploads/clip.mp4 must not share or delete a copy);
        # a changed file (new mtime) gets a new copy
        prefix = f"{path.stem}_{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]}_"
        dst = PREVIEW_DIR / f"{prefix}{int(mtime(path))}.mp4"
        if dst.is_file():
            if video_codec(dst) in BROWSER_CODECS:                     # a finished, playable copy
                return dst
            dst.unlink()                                               # a half-written copy (the app was stopped)
        part = dst.with_name(dst.stem + ".part.mp4")                   # convert under a temporary name ...
        if not transcode_h264(path, part):
            part.unlink(missing_ok=True)
            return None
        os.replace(part, dst)                                          # ... and only then give it the real name
        for old in PREVIEW_DIR.glob(f"{glob.escape(prefix)}*.mp4"):     # older copies of this same file only
            if old != dst and old.stem[len(prefix):].isdigit():
                old.unlink(missing_ok=True)
        return dst
    except Exception:                                       # a preview must never take the page down
        return None


def draw_zones(frame_bgr, zones: list[dict], scale: float = 1.0):
    """Draw zone polygons (points in ORIGINAL video pixels) on a copy of the frame; returns an RGB array."""
    import cv2
    import numpy as np
    img = frame_bgr.copy()
    polys = []
    for z in zones:
        raw = np.array([[x * scale, y * scale] for x, y in z["points"]], dtype=float)
        # A NaN, infinite or absurdly large point must not crash the picture: clamp it far outside the frame.
        pts = np.clip(np.nan_to_num(raw, nan=0.0, posinf=MAX_COORD, neginf=-MAX_COORD), -MAX_COORD, MAX_COORD).astype(np.int32)
        if len(pts) >= 3:
            polys.append((z.get("name") or "zone", pts))
    if polys:
        shade = img.copy()
        for _name, pts in polys:
            cv2.fillPoly(shade, [pts], ZONE_BGR)
        img = cv2.addWeighted(shade, 0.25, img, 0.75, 0)           # translucent fill
        for name, pts in polys:
            cv2.polylines(img, [pts], True, ZONE_BGR, 2, cv2.LINE_AA)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            cv2.putText(img, str(name).encode("ascii", "replace").decode("ascii"), (x + 4, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ZONE_BGR, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- zones

def rect_zone(x1: float, y1: float, x2: float, y2: float, name: str = "restricted") -> list[dict]:
    """One rectangular zone from two opposite corners (any order), points in original video pixels."""
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    return [{"name": name, "points": [(left, top), (right, top), (right, bottom), (left, bottom)]}]


def rect_is_empty(zones: list[dict], min_side: float = 2.0) -> bool:
    """True when the rectangle is too thin to be a zone (both sliders on the same value)."""
    xs = [p[0] for p in zones[0]["points"]]
    ys = [p[1] for p in zones[0]["points"]]
    return (max(xs) - min(xs)) < min_side or (max(ys) - min(ys)) < min_side


def write_zone_file(stem: str, zones: list[dict], image_size) -> Path:
    """Write zones to outputs/_ui/<video stem>_zones.json (standard zones.json format) and return the path."""
    path = UI_DIR / f"{safe_stem(stem, 'zones')}_zones.json"
    utils.save_zones(path, zones, image_size)
    return path


ZONE_LAYOUT_HELP = ('expected {"zones": [{"name": "restricted", "points": [[x, y], [x, y], [x, y], ...]}]} '
                    "(see zones.example.json)")


def friendly_zone_error(exc: Exception) -> str:
    """A zones.json problem in words a person can act on (not "'str' object has no attribute 'get'")."""
    text = str(exc)
    if isinstance(exc, (AttributeError, TypeError, KeyError)):
        return "the file has an unexpected layout; " + ZONE_LAYOUT_HELP
    if isinstance(exc, OverflowError):
        return f"a number in the file is far too large (limit {MAX_COORD:,})"
    if isinstance(exc, ValueError) and text.startswith("could not convert"):
        return "a point or the image_size is not a number; " + ZONE_LAYOUT_HELP
    return text


def zones_problem(zones: list[dict]) -> str:
    """"" when every zone point is a finite number of sane size, else a sentence saying what is wrong."""
    for z in zones:
        for x, y in z["points"]:
            if not (math.isfinite(x) and math.isfinite(y)):
                return f"zone '{z['name']}' has a point that is not a finite number"
            if abs(x) > MAX_COORD or abs(y) > MAX_COORD:
                return f"zone '{z['name']}' has a point further than {MAX_COORD:,} pixels from the frame"
    return ""


def zones_outside_frame(zones: list[dict], size) -> bool:
    """True when there are zones and none of them touches the frame (so nothing could ever enter them)."""
    if not zones:
        return False
    width, height = float(size[0]), float(size[1])
    for z in zones:
        xs = [p[0] for p in z["points"]]
        ys = [p[1] for p in z["points"]]
        if max(xs) >= 0 and min(xs) <= width and max(ys) >= 0 and min(ys) <= height:
            return False
    return True


def save_uploaded_zones(stem: str, data: bytes) -> tuple[Path | None, str]:
    """Store an uploaded zones.json under outputs/_ui/ after checking it holds a zone. Returns (path, error)."""
    try:
        text = bytes(data).decode("utf-8-sig")
        json.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:
        return None, f"not valid JSON ({exc})"
    UI_DIR.mkdir(parents=True, exist_ok=True)
    path = UI_DIR / f"{safe_stem(stem, 'zones')}_uploaded_zones.json"
    path.write_text(text, encoding="utf-8")
    try:
        zones = utils.load_zones(path, (640, 360))       # a dummy size, so an invalid image_size is caught too
    except (ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError, OverflowError) as exc:
        path.unlink(missing_ok=True)
        return None, friendly_zone_error(exc)
    if not zones:
        path.unlink(missing_ok=True)
        return None, "it contains no zone with 3 or more points"
    problem = zones_problem(zones)
    if problem:
        path.unlink(missing_ok=True)
        return None, problem
    return path, ""


def zones_in_video_pixels(path, size) -> tuple[list[dict], str]:
    """Zones of a zones.json scaled to a video of `size` = (W, H) pixels, for drawing. Returns (zones, error).

    A file with an "image_size" is scaled by size / image_size (what run.py does). A bare list of points has
    no size and means "processing pixels", so it is scaled up from the default 640 px processing width.
    """
    try:
        data = utils.read_json(path)
        width, height = int(size[0]), int(size[1])
        if isinstance(data, dict) and data.get("image_size"):
            zones = utils.load_zones(path, (width, height))
        else:
            proc_w = int(utils.load_config(None)["video"]["resize_width"] or width)
            factor = width / float(proc_w)
            zones = [{"name": z["name"], "points": [(x * factor, y * factor) for x, y in z["points"]]}
                     for z in utils.load_zones(path, None)]
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError, OverflowError) as exc:
        return [], f"cannot read {rel_path(path)}: {friendly_zone_error(exc)}"
    problem = zones_problem(zones)
    if problem:
        return [], f"{rel_path(path)}: {problem}"
    return zones, ""


# --------------------------------------------------------------------------- the run.py command

def default_out_dir(video: Path, scenario: str | None) -> str:
    """outputs/<video stem>, plus _<scenario> for every preset except campus (the same rule as run.py)."""
    suffix = f"_{scenario}" if scenario and scenario != "campus" else ""
    return f"outputs/{Path(video).stem}{suffix}"


def build_command(video, scenario: str, out_dir: str, zones=None, privacy: bool = False, device: str = "auto",
                  stride: int | None = None, max_seconds: float = 0.0, reuse: bool = True,
                  no_video: bool = False, pose: bool = False, start_time: str | None = None,
                  camera_name: str | None = None) -> list[str]:
    """The exact `run.py` command line for the chosen options (a list for subprocess; no shell involved).

    Flags equal to run.py's own defaults are left out, so the command in the log reads like one a person would type.
    """
    cmd = [sys.executable, "aquawatch/run.py", "--video", rel_path(video), "--scenario", scenario, "--out", out_dir]
    if zones:
        cmd += ["--zones", rel_path(zones)]
    if privacy:                           # faces are not blurred by default (config.yaml privacy.enabled)
        cmd.append("--privacy")
    if device and device != "auto":
        cmd += ["--device", str(device)]
    if stride is not None:
        cmd += ["--stride", str(int(stride))]
    if max_seconds and float(max_seconds) > 0:
        cmd += ["--max-seconds", f"{float(max_seconds):g}"]
    if pose:
        cmd.append("--pose")
    if no_video:
        cmd.append("--no-video")
    if start_time and str(start_time).strip():
        cmd += ["--start-time", str(start_time).strip()]
    if camera_name and str(camera_name).strip():
        cmd += ["--camera-name", str(camera_name).strip()]
    if reuse:
        cmd.append("--reuse")
    return cmd


def display_command(cmd: list[str]) -> str:
    """The command as a person would type it: the interpreter shown as `python`, arguments quoted when needed."""
    return subprocess.list2cmdline(["python"] + [str(c) for c in cmd[1:]])


# --------------------------------------------------------------------------- cached detections (--reuse)

def _detections_match(header: dict, video: Path, classes: list[int], stride: int, size, resize_width=None,
                      weights: str = "", max_seconds: float = 0.0, n_frames: int = 0, weights_sha1: str = "",
                      cfg: dict | None = None) -> bool:
    """Same checks as run.load_reusable_detections: video name, stride, classes, frame size, resize width,
    detector and how much of the video the cached run covered (max seconds / partial runs).

    The extra arguments are optional; a check whose value is left empty is skipped.
    """
    try:
        if Path(str(header.get("video", ""))).name != Path(video).name:
            return False
        if int(header.get("stride", 0)) != int(stride):
            return False
        if sorted(int(c) for c in (header.get("classes") or [0])) != sorted(int(c) for c in classes):
            return False
        orig = header.get("orig_size")
        if size and orig and [int(v) for v in orig] != [int(v) for v in size]:
            return False
        proc = header.get("proc_size")
        if resize_width is not None and proc:                          # 0 means "keep the original width"
            want_width = int(resize_width) or (int(orig[0]) if orig else 0)
            if want_width and int(proc[0]) != want_width:
                return False
        if weights and not utils.same_detector(header.get("model"), header.get("model_sha1"), weights, weights_sha1):
            return False                                               # other detector, or the same name replaced
        fps = float(header.get("fps") or 0.0) or 30.0
        covered = float(header.get("duration_s") or 0.0)
        if max_seconds > 0:
            if covered > max_seconds + 2.0 / fps:                      # the cached run covers more than asked
                return False
            available = (n_frames / fps) if n_frames else max_seconds
            if covered < min(max_seconds, available) - 2.0 / fps:      # ... or less than asked
                return False
        read = int(header.get("n_frames_read") or 0)
        if max_seconds == 0 and n_frames and read and read < 0.98 * n_frames - 2:
            return False                                               # the cached run covers only part of the video
        if max_seconds == 0 and n_frames and read and read > n_frames + 2:
            return False                                               # longer than the video: the video changed
        if cfg is not None and utils.tracking_settings_mismatch(header, cfg):
            return False                                               # other conf / iou / imgsz / tracker yaml
    except (TypeError, ValueError, IndexError):                        # a header with odd values is not reusable
        return False
    return True


def find_cached_detections(video, out_dir, classes: list[int], stride: int, size=None, files_signature=None,
                           resize_width=None, weights: str = "", max_seconds: float = 0.0,
                           n_frames: int = 0, weights_sha1: str = "", preset: str | None = None) -> str | None:
    """Path of the detections.json that `--reuse` can use: the one in `out_dir`, else one from another outputs/ folder.
    With `preset`, the detector / tracker settings (conf, iou, imgsz, tracker yaml) must match too (run.py's rule).

    `files_signature` only exists so a caller can cache this function: pass something that changes when the
    detections files change (see detections_signature). Returns None when nothing compatible exists.
    """
    video = Path(video)
    cfg = None
    if preset:
        try:
            cfg = utils.load_config(None, preset)
        except Exception:                                    # a broken preset: no cache can be trusted
            return None
    own = ROOT / out_dir / "detections.json"
    candidates = [own] if own.is_file() else []
    others = [p for p in _detections_files() if p != own]
    candidates += sorted(others, key=mtime, reverse=True)
    for cand in candidates:
        try:
            header = utils.read_json(cand)
        except (OSError, ValueError):
            continue
        if isinstance(header, dict) and _detections_match(header, video, classes, stride, size, resize_width,
                                                          weights, max_seconds, n_frames, weights_sha1, cfg):
            return str(cand)
    return None


def weights_present(name: str) -> bool:
    """Is the detector's weights file in the project folder? `name` is the configured path ("yolo11n.pt" or
    "training/runs/swimmer/weights/best.pt", or absolute). If it is missing, Ultralytics tries to download it."""
    return utils.weights_file(name) is not None


def weights_missing_note(path: str) -> str:
    """Warning for a detector weights file that is not on disk ("" when it is there). Ultralytics only downloads
    its official models (a bare name like yolo11n.pt); a trained model in a folder (best.pt) must be copied in."""
    if weights_present(path):
        return ""
    p = Path(str(path))
    if p.parent == Path("."):
        return f"`{p.name}` is not in the project folder: Ultralytics would try to download it, which needs internet."
    return (f"`{p.name}` is not in the project folder (expected at `{p.as_posix()}`). It cannot be downloaded by "
            "Ultralytics, so the run would fail: run `python aquawatch/get_models.py` once to get the models.")


def yolo_seconds(facts: dict, stride: int, max_seconds: float = 0.0, frames_per_s: float = 5.5) -> float:
    """Rough time YOLO + ByteTrack need on a laptop CPU (about 5.5 processed frames per second); 0 when unknown."""
    frames = int(facts.get("frames") or 0)
    if max_seconds and float(max_seconds) > 0 and facts.get("fps"):
        frames = min(frames, int(float(max_seconds) * float(facts["fps"]))) if frames else 0
    return frames / max(int(stride), 1) / frames_per_s if frames else 0.0


def _detections_files() -> list[Path]:
    """Every detections.json that could be reused: outputs/*/detections.json and the copies kept in outputs/_ui/cache/."""
    return sorted(OUTPUTS_DIR.glob("*/detections.json")) + sorted(CACHE_DIR.glob("*.json"))


def detections_signature() -> tuple:
    """(path, mtime) of every reusable detections file: a cheap cache key for find_cached_detections."""
    return tuple((str(p), mtime(p)) for p in _detections_files())


def preserve_detections(out_dir, reuse_from: str | None = None) -> str:
    """Keep a copy of `out_dir`/detections.json before a run replaces it. Returns a log line ("" when nothing was kept).

    Expensive YOLO results must survive a run with another scenario: picking the wrong preset for
    samples/clip.mp4 would otherwise overwrite the detections that make the right preset take seconds.
    The copy goes to outputs/_ui/cache/<folder name>.json and is found again by find_cached_detections.
    """
    own = ROOT / out_dir / "detections.json"
    if not own.is_file() or (reuse_from and Path(reuse_from) == own):
        return ""                                               # nothing there, or this run keeps using it
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dest = CACHE_DIR / f"{Path(out_dir).name}.json"
        if reuse_from and Path(reuse_from).resolve() == dest.resolve():
            # this run is about to reuse that very cache file: keep the old detections under another name, named
            # by their contents so switching a setting back and forth reuses one copy instead of piling up new ones
            dest = CACHE_DIR / f"{Path(out_dir).name}_{hashlib.sha1(own.read_bytes()).hexdigest()[:10]}.json"
        shutil.copy2(own, dest)
    except OSError:
        return ""
    return f"[app] kept the old detections of {rel_path(own)} as {rel_path(dest)}; this run replaces them"


def seed_detections(found: str | None, out_dir) -> str:
    """Copy a compatible detections.json into `out_dir` so `run.py --reuse` finds it there. Returns a log line."""
    if not found:
        return ""
    target = ROOT / out_dir / "detections.json"
    if Path(found) == target:
        return f"[app] reusing cached detections in {rel_path(target)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(found).read_bytes())
    kps = Path(found).parent / "keypoints.json"                 # pool scenario: reuse the pose keypoints too
    if kps.is_file():                                            # they belong to the detections just copied, so they
        (target.parent / "keypoints.json").write_bytes(kps.read_bytes())   # replace any older ones (pool.py checks the key)
    return f"[app] copied cached detections from {rel_path(found)} into {rel_path(target)}"


# --------------------------------------------------------------------------- running run.py and reading its output

STEP_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.*?)(?:\s+\.\.\.)?\s*$")      # "[3/9] Detecting behaviours (...) ..."
PCT_RE = re.compile(r"^\s*tracking\s+(\d+)%")                           # "  tracking  45%  (frame ..."


class Progress:
    """Turns run.py's "[k/N] step ..." lines (and the tracker's "tracking 45%") into a 0..1 fraction and a label."""

    def __init__(self, total_steps: int = 8):
        self.total = total_steps
        self.step = 0
        self.inside = 0.0                          # progress inside the current step, 0..1
        self.label = "Starting run.py"

    @property
    def fraction(self) -> float:
        """Finished steps plus the part of the current one that is done, over all steps."""
        return min(1.0, max(0.0, (max(self.step - 1, 0) + self.inside) / max(self.total, 1)))

    def feed(self, line: str) -> float | None:
        """Update from one output line; returns the new fraction, or None when the line says nothing about progress."""
        m = STEP_RE.match(line)
        if m:
            self.step, self.total = int(m.group(1)), int(m.group(2))
            self.inside = 0.0
            self.label = f"[{self.step}/{self.total}] {m.group(3)}"
            return self.fraction
        m = PCT_RE.match(line)
        if m:
            self.inside = min(1.0, int(m.group(1)) / 100.0)
            return self.fraction
        return None


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Stop a process AND the programs it started (run.py starts ffmpeg), so nothing keeps working in the background."""
    if proc.poll() is None and os.name == "nt":
        try:                                                    # /T = the whole tree, /F = force
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            pass
    elif proc.poll() is None:                                   # Linux / macOS: the child leads its own process group
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            pass
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


# --------------------------------------------------------------------------- a new multi-camera site

SITE_VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
SITES_DATA_DIR = ROOT / "data" / "sites"
# rough seconds of work per second of video on this laptop (2 CPU threads, detector + pose + drawing), measured on
# 2026-10-12: Wide CCTV ~30, the fast models ~7. Only for the estimate shown before a run.
SECONDS_PER_VIDEO_SECOND = {"pool_hd": 30.0, "pool_trained": 7.0, "pool": 8.0}
SITE_KINDS = {"Real CCTV or phone recordings": "REAL multi-camera footage (added from the dashboard)",
              "Synthetic or rendered videos": "SYNTHETIC multi-camera footage (added from the dashboard)",
              "Crops of one recording": "VIRTUAL multi-camera footage: crops of one recording (added from the dashboard)"}


def site_folder_videos(folder) -> list[Path]:
    """The videos directly inside a folder (relative paths are taken from the project folder), sorted by name."""
    base = Path(str(folder or "").strip().strip('"'))
    if not str(base) or str(base) == ".":
        return []
    base = base if base.is_absolute() else ROOT / base
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in SITE_VIDEO_TYPES)


def site_slug(name: str) -> str:
    """'Main pool, Saturday' -> 'main_pool_saturday' (a folder name for outputs/ and data/sites/)."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:40]


def display_name(video) -> str:
    """A video's file name as a readable title: no folder, no extension, '_' as spaces
    ('WhatsApp_Video_2026-10-07_at_15.59.14.mp4' -> 'WhatsApp Video 2026-10-07 at 15.59.14')."""
    name = Path(str(video or "").replace("\\", "/")).name
    stem = name[: -len(Path(name).suffix)] if Path(name).suffix.lower() in VIDEO_EXTS else name
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip() or name


def short_camera_name(video) -> str:
    """A readable default camera name from a file name: its END (where clips of one folder differ, e.g.
    'WhatsApp_Video_2026-10-09_at_07.21.29' -> '2026-10-09_at_07.21.29'), at most 24 characters."""
    stem = Path(str(video)).stem
    if len(stem) <= 24:
        return stem
    tail = stem[-24:]
    if stem[-25] not in "_-. ":                      # the cut fell inside a word: start after the next separator
        cuts = [tail.find(sep) for sep in "_- ." if 0 <= tail.find(sep) < len(tail) - 1]
        tail = tail[min(cuts) + 1:] if cuts else tail
    return tail.lstrip("_-. ")


def build_site_registry(name: str, kind: str, preset: str, cameras: list[dict]) -> dict:
    """The cameras.yaml content of a new site. cameras: [{"video", "name", "start_time"}] (start time 'YYYY-MM-DD
    HH:MM:SS'). Clocks typed by a person are 'user_declared' (not verified), so cross-camera answers say so."""
    rows = []
    for n, cam in enumerate(cameras, start=1):
        label = str(cam.get("name") or "").strip()
        if not re.match(r"(?i)cam(era)?\s*\d", label):             # 'CAM01 <name>': numbered in the CHOSEN order
            label = f"CAM{n:02d} {label}".strip()
        rows.append({"id": f"cam{n:02d}", "name": label,
                     "video": rel_path(cam["video"]), "start_time": str(cam["start_time"]).strip(), "preset": preset,
                     "offset_s": 0.0, "sync_status": "user_declared"})
    return {"site": str(name).strip(), "label": SITE_KINDS.get(kind, kind), "cameras": rows}


def site_problems(name: str, cameras: list[dict]) -> list[str]:
    """Why a new site cannot be made yet (empty list = fine)."""
    problems = []
    if not site_slug(name):
        problems.append("Give the site a name.")
    elif (OUTPUTS_DIR / site_slug(name) / "aquawatch.sqlite3").exists():
        problems.append(f"A site called '{site_slug(name)}' already exists: choose another name.")
    if not cameras:
        problems.append("Pick at least one video.")
    from cameras import parse_wall
    for cam in cameras:
        try:
            parse_wall(str(cam.get("start_time")))
        except ValueError:
            problems.append(f"Start time of {Path(str(cam.get('video'))).name} must look like 2026-10-12 15:00:00.")
    return problems


def write_site(registry: dict) -> Path:
    """Write data/sites/<slug>/cameras.yaml (never over an existing one). Returns its path."""
    import yaml
    folder = SITES_DATA_DIR / site_slug(registry["site"])
    path = folder / "cameras.yaml"
    if path.exists():
        raise FileExistsError(f"{rel_path(path)} already exists")
    folder.mkdir(parents=True, exist_ok=True)
    header = ("# Made from the AquaWatch dashboard. Start times were typed by a person: sync_status user_declared.\n"
              "# Add zones per camera with   zones: <file>.json   (aquawatch/zone_picker.py) and re-run aquawatch/ingest.py.\n")
    path.write_text(header + yaml.safe_dump(registry, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def site_ingest_command(cameras_yaml, name: str, privacy: bool = False) -> list[str]:
    """The ingest.py command for a new site: results in outputs/<slug>/ and the index outputs/<slug>/aquawatch.sqlite3."""
    slug = site_slug(name)
    cmd = [sys.executable, "-B", "aquawatch/ingest.py", "--cameras", rel_path(cameras_yaml),
           "--db", f"outputs/{slug}/aquawatch.sqlite3", "--out-root", f"outputs/{slug}"]
    return cmd + (["--privacy"] if privacy else [])


def site_estimate_minutes(durations_s: list[float], preset: str) -> float:
    """Rough minutes for the whole ingest on this laptop (one camera after the other)."""
    return sum(durations_s) * SECONDS_PER_VIDEO_SECOND.get(preset, 30.0) / 60.0


def run_pipeline(cmd: list[str], on_line=None, cwd=None, on_idle=None, idle_s: float = 0.5) -> tuple[int, list[str]]:
    """Run a command, merge stderr into stdout, call `on_line(text)` for every line as it arrives.

    `on_idle()` is called every `idle_s` seconds while the command is silent (the page uses it to show the
    elapsed time; it also lets Streamlit notice that the user changed a widget and stop this run).
    Returns (exit code, the last MAX_LOG_LINES lines, each cut to MAX_LINE_CHARS). Never raises for a failing
    command: a process that cannot start gives exit code 1 and a one-line explanation. The child (and its own
    children) is killed if this function is left early, so a long YOLO job is never left running in the background.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd or ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                start_new_session=(os.name != "nt"))      # own process group, so it can be killed whole
    except OSError as exc:
        return 1, [f"could not start the process: {exc}"]

    inbox: queue.Queue = queue.Queue()

    def pump() -> None:
        """Reader thread: move output lines into the queue so the main loop can also wake up while it is silent."""
        try:
            for raw in proc.stdout:
                inbox.put(raw)
        except (OSError, ValueError):                           # the pipe was closed under us
            pass
        finally:
            inbox.put(None)                                     # marks the end of the output

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    lines: deque = deque(maxlen=MAX_LOG_LINES)
    try:
        while True:
            try:
                raw = inbox.get(timeout=idle_s)
            except queue.Empty:
                if on_idle is not None:
                    on_idle()
                continue
            if raw is None:
                break
            line = raw.rstrip("\r\n")[:MAX_LINE_CHARS]
            lines.append(line)
            if on_line is not None:
                on_line(line)
        return proc.wait(), list(lines)
    finally:
        kill_process_tree(proc)                                 # does nothing when the process has ended
        reader.join(timeout=5)
        if proc.stdout and not reader.is_alive():               # never close a pipe another thread is still reading
            proc.stdout.close()


def last_lines(lines: list[str], n: int = 20, max_chars: int = 400) -> str:
    """The last `n` non-empty lines joined with newlines, each cut to `max_chars` (for the error box)."""
    return "\n".join([ln[:max_chars] for ln in lines if ln.strip()][-n:])


# --------------------------------------------------------------------------- remembering the last run

def save_last_run(info: dict) -> None:
    """Write the latest run (command, log, results folder) to outputs/_ui/last_run.json. Never raises."""
    try:
        utils.write_json(LAST_RUN_FILE, {"when": time.strftime("%Y-%m-%d %H:%M:%S"), **info})
    except (OSError, TypeError, ValueError):
        pass


def load_last_run() -> dict | None:
    """The run saved by save_last_run, or None when there is none, it is unreadable, or its results are gone.

    A browser reload starts a new Streamlit session, so without this file the results would disappear.
    """
    try:
        info = utils.read_json(LAST_RUN_FILE)
        out_dir = str(info["out_dir"])
        (ROOT / out_dir).resolve().relative_to(ROOT.resolve())  # a results folder outside the project is not restored
        ok = bool(info["ok"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if ok and not (ROOT / out_dir / "events.json").is_file():
        return None
    lines = info.get("lines")
    return {"ok": ok, "code": int(as_float(info.get("code"), -1)), "out_dir": out_dir,
            "cmd": str(info.get("cmd") or ""), "interpreter": str(info.get("interpreter") or ""),
            "lines": [str(x) for x in lines][-400:] if isinstance(lines, list) else [],
            "seconds": as_float(info.get("seconds"), 0.0), "video": str(info.get("video") or ""),
            "scenario": str(info.get("scenario") or ""), "when": str(info.get("when") or "")}


# --------------------------------------------------------------------------- reading a results folder

def as_float(value, default=None):
    """`value` as a finite float, or `default` when it is missing, not a number, NaN or infinite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _dicts(value) -> list[dict]:
    """The dict items of a list ([] when `value` is not a list): events.json may come from an older version."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _subdict(value) -> dict:
    """`value` when it is a dict, else {}."""
    return value if isinstance(value, dict) else {}


def _event_ids(item: dict) -> list[int]:
    """The event ids of an incident chain as integers (anything that is not a number is skipped)."""
    ids = item.get("event_ids")
    numbers = [as_float(i) for i in ids] if isinstance(ids, list) else []
    return [int(n) for n in numbers if n is not None]


def sanitize_results(data) -> dict:
    """events.json content in a safe shape for the page: right container types, numbers as numbers.

    Raises ValueError when the file does not hold a JSON object. Everything else is repaired quietly (a wrong
    type becomes an empty one), so an old or hand-edited file shows what it can instead of crashing the page.
    """
    if not isinstance(data, dict):
        raise ValueError("events.json does not hold an object")
    out = dict(data)
    meta = dict(_subdict(data.get("meta")))
    cfg = _subdict(meta.get("config"))
    scenario = _subdict(cfg.get("scenario"))
    classes = _subdict(cfg.get("model")).get("classes")
    weights = _subdict(_subdict(cfg.get("events")).get("confidence_weights"))
    meta["config"] = {                                          # only the parts the page reads, each in a safe type
        "scenario": {"entity_word": str(scenario.get("entity_word") or "Person"),
                     "behavior_names": {str(k): str(v) for k, v in _subdict(scenario.get("behavior_names")).items()}},
        "model": {"classes": [int(c) for c in map(as_float, classes) if c is not None] or [0]
                  if isinstance(classes, list) else [0]},
        "events": {"confidence_weights": {k: as_float(v) for k, v in weights.items() if as_float(v) is not None}}}
    meta["scenario"] = {"name": str(_subdict(meta.get("scenario")).get("name") or ""),
                        "title": str(_subdict(meta.get("scenario")).get("title") or "")}
    meta["duration_s"] = as_float(meta.get("duration_s"))
    meta["processing_fps"] = as_float(meta.get("processing_fps"))
    n_tracks = as_float(meta.get("n_tracks"))
    meta["n_tracks"] = int(n_tracks) if n_tracks is not None else None
    out["meta"] = meta

    events = []
    for e in _dicts(data.get("events")):
        e = dict(e)
        e["confidence"] = as_float(e.get("confidence"), 0.0)
        e["duration_s"] = as_float(e.get("duration_s"))
        parts = _subdict(e.get("confidence_parts"))
        e["confidence_parts"] = {k: as_float(v) for k, v in parts.items() if as_float(v) is not None}
        e["baseline"] = _subdict(e.get("baseline"))
        pose = dict(_subdict(e.get("pose")))
        for key in ("median_lean_deg", "max_leg_spread_bh"):
            pose[key] = as_float(pose.get(key))
        e["pose"] = pose
        for key in ("evidence", "entity_name", "behavior_name", "zone", "start", "end", "snapshot", "speed_plot"):
            if e.get(key) is not None and not isinstance(e[key], str):
                e[key] = str(e[key])
        events.append(e)
    out["events"] = events
    out["incidents"] = [dict(inc, event_ids=_event_ids(inc)) for inc in _dicts(data.get("incidents"))]
    out["near_misses"] = _dicts(data.get("near_misses"))
    out["unusual_tracks"] = _dicts(data.get("unusual_tracks"))
    reel = data.get("highlight_reel")
    out["highlight_reel"] = reel if isinstance(reel, str) else None
    return out


def incident_items(data: dict) -> list[dict]:
    """The incidents the way the highlight reel counts them: every chain, plus each event that is not in a chain."""
    chains = _dicts(data.get("incidents"))
    chained = {i for inc in chains for i in _event_ids(inc)}
    singles = [e for e in _dicts(data.get("events")) if as_float(e.get("event_id")) not in chained]
    return chains + singles


def incident_count(data: dict) -> int:
    """Number of incidents (see incident_items)."""
    return len(incident_items(data))


def tracked_label(cfg: dict | None) -> str:
    """Plural noun for what the scenario tracks: "People", "Animals", or "Objects" when vehicles are tracked too."""
    cfg = _subdict(cfg)
    word = str(utils.entity_word({"scenario": _subdict(cfg.get("scenario"))})).lower()
    classes = _subdict(cfg.get("model")).get("classes")
    if word == "person" and len(classes if isinstance(classes, list) and classes else [0]) > 1:
        word = "object"
    return "People" if word == "person" else word.capitalize() + "s"


def high_incident_count(data: dict) -> int:
    """How many of the incidents have severity high."""
    return sum(1 for item in incident_items(data) if str(item.get("severity")).lower() == "high")


def result_metrics(data: dict) -> dict:
    """The numbers for the metric tiles of a results folder (events.json content)."""
    meta = _subdict(data.get("meta"))
    cfg = _subdict(meta.get("config"))
    scenario = _subdict(meta.get("scenario"))
    duration = as_float(meta.get("duration_s"))
    fps = as_float(meta.get("processing_fps"))
    return {
        "incidents": incident_count(data),
        "high": high_incident_count(data),
        "events": len(_dicts(data.get("events"))),
        "tracked_label": tracked_label(cfg),
        "tracked": meta.get("n_tracks"),
        "duration": utils.fmt_time(duration) if duration is not None else "?",
        "processing_fps": f"{fps:.1f}" if fps else None,
        "scenario": str(scenario.get("name") or "?"),
        "scenario_title": str(scenario.get("title") or ""),
        "privacy": bool(meta.get("privacy")),
    }


def confidence_weights(data: dict) -> dict:
    """Weights of margin / duration / detection from the run's own config (defaults when absent)."""
    cfg = _subdict(_subdict(data.get("meta")).get("config"))
    weights = _subdict(_subdict(cfg.get("events")).get("confidence_weights"))
    return {"margin": as_float(weights.get("margin"), 0.4), "duration": as_float(weights.get("duration"), 0.3),
            "detection": as_float(weights.get("detection"), 0.3)}


def confidence_line(event: dict, weights: dict) -> str:
    """"confidence 0.78 = 0.4 x 0.81 + 0.3 x 0.62 + 0.3 x 0.89": the score and its three parts."""
    conf = as_float(event.get("confidence"), 0.0)
    parts = _subdict(event.get("confidence_parts"))
    terms = [f"{weights[k]:g} x {as_float(parts[k]):.2f}" for k in ("margin", "duration", "detection")
             if as_float(parts.get(k)) is not None]
    return f"confidence {conf:.2f} = " + " + ".join(terms) if terms else f"confidence {conf:.2f}"


def severity_color(severity) -> str:
    """Streamlit badge colour for high / medium / low (gray when unknown)."""
    return SEVERITY_COLORS.get(str(severity).lower(), "gray")


def md_escape(text) -> str:
    """Escape characters that Streamlit's Markdown would treat as formatting, so evidence text shows literally."""
    return re.sub(r"([\\`*_\[\]<>$~|])", r"\\\1", str(text))


def pose_text(event: dict) -> tuple[str, str] | None:
    """(colour, sentence) for the pose verdict of an event, or None when the pose check did not run."""
    verified = event.get("verified")
    if verified is None:
        return None
    pose = _subdict(event.get("pose"))
    bits = []
    lean, spread = as_float(pose.get("median_lean_deg")), as_float(pose.get("max_leg_spread_bh"))
    if lean is not None:
        bits.append(f"median lean {lean:.1f} deg")
    if spread is not None:
        bits.append(f"max leg spread {spread:.2f} bh")
    if pose.get("usable_frames") is not None:
        bits.append(f"{pose['usable_frames']}/{pose.get('frames_checked', '?')} usable frames")
    detail = (" (" + ", ".join(bits) + ")") if bits else ""
    if verified is True:
        return "green", "Pose check agrees" + detail
    if verified is False:
        return "red", "Pose check disagrees" + detail
    return "gray", "Pose check uncertain" + detail
