"""AquaWatch command-line entry point: a video goes in, an evidence report comes out.

    python aquawatch/run.py --video samples/clip.mp4 --scenario pool_trained   # close or medium shot
    python aquawatch/run.py --video samples/clip.mp4 --scenario pool_hd        # wide, crowded CCTV (slower)
    python aquawatch/run.py --video samples/clip.mp4 --scenario pool --reuse   # skip YOLO, re-run only the rules
    python aquawatch/run.py --video samples/clip.mp4 --privacy                 # blur faces in every output
    python aquawatch/run.py --list-scenarios                                   # show the presets and exit
    python aquawatch/run.py --webcam 30                                        # record 30 s, then analyse

The pipeline (see docs/DESIGN.md):
    tracker (YOLO + ByteTrack) -> features -> behaviors -> events -> [pose check] -> chains (incidents)
    -> render -> highlights -> report

`tracker`, `render`, `highlights` and `pose_verify` are imported lazily inside the functions that need
them, so `--reuse` (rules only) works on a machine that has no ultralytics / torch installed.
"""
from __future__ import annotations

import argparse
import json
import contextlib
import datetime as _dt
import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

import utils
from behaviors import detect_behaviors
from events import build_events
from features import build_tracks, track_summary
from stitch import drop_deck_tracks, stitch_tracklets
from report import write_outputs


# --------------------------------------------------------------------------- small console helpers

def ascii_safe(text: str) -> str:
    """Replace any non-ASCII character so a Windows console (cp1252 etc.) never crashes."""
    return str(text).encode("ascii", "replace").decode("ascii")


def fail(message: str, code: int = 2):
    """Print a one-line error and stop. Used for problems the user can fix (bad path, no camera...)."""
    print(f"error: {ascii_safe(message)}", file=sys.stderr)
    sys.exit(code)


def warn(message: str) -> None:
    """Print a warning that does not stop the run."""
    print(f"warning: {ascii_safe(message)}", file=sys.stderr, flush=True)


class Steps:
    """Prints '[n/N] name ...' before each pipeline step and its elapsed time afterwards."""

    def __init__(self, total: int):
        self.total = total
        self.count = 0
        self.timings: dict[str, float] = {}

    @contextlib.contextmanager
    def step(self, name: str):
        self.count += 1
        print(f"[{self.count}/{self.total}] {name} ...", flush=True)
        t0 = time.perf_counter()
        yield
        dt = time.perf_counter() - t0
        self.timings[name] = dt
        print(f"      done in {dt:.1f} s", flush=True)


# --------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    """Define the CLI of docs/DESIGN.md sections 3.8 and 4.5 (--scenario, --list-scenarios, --privacy)."""
    p = argparse.ArgumentParser(
        prog="run.py",
        description="AquaWatch: find swimmers who may be in distress in a pool video and explain each alert "
                    "with evidence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python aquawatch/run.py --video samples/clip.mp4 --scenario pool_trained
              python aquawatch/run.py --video samples/clip.mp4 --scenario pool_hd --zones zones.json
              python aquawatch/run.py --video samples/clip.mp4 --scenario pool --reuse   (after editing config.yaml)
              python aquawatch/run.py --video samples/clip.mp4 --privacy   (faces blurred in every output)
              python aquawatch/run.py --list-scenarios
              python aquawatch/run.py --webcam 30
            """),
    )
    src = p.add_mutually_exclusive_group()              # one of them is required, except with --list-scenarios
    src.add_argument("--video", metavar="PATH", help="input video file")
    src.add_argument("--webcam", type=float, metavar="SECONDS",
                     help="record SECONDS from the laptop camera to samples/webcam_<n>.mp4, then analyse it")
    p.add_argument("--zones", metavar="zones.json", help="pool zone polygons (make one with aquawatch/zone_picker.py)")
    p.add_argument("--config", metavar="config.yaml", help="settings file (default: config.yaml next to run.py)")
    p.add_argument("--out", metavar="DIR", help="output folder (default: outputs/<video name>)")
    p.add_argument("--device", metavar="auto|cpu|0", help="where YOLO runs (default: from config.yaml)")
    p.add_argument("--stride", type=int, metavar="N", help="process every Nth frame (default: from config.yaml)")
    p.add_argument("--max-seconds", type=float, metavar="S", help="only analyse the first S seconds")
    p.add_argument("--pose", action="store_true", help="second opinion with a pose model (running / loitering)")
    p.add_argument("--no-video", action="store_true", help="skip annotated.mp4 and highlights.mp4 (faster)")
    p.add_argument("--reuse", action="store_true",
                   help="reuse <out>/detections.json if it matches (skips YOLO; fast threshold tuning)")
    p.add_argument("--scenario", metavar="NAME",
                   help="preset from scenarios/ (see --list-scenarios): which classes, rules and wording to use")
    p.add_argument("--list-scenarios", action="store_true", help="print the available presets and exit")
    face = p.add_mutually_exclusive_group()
    face.add_argument("--privacy", action="store_true",
                      help="blur faces in annotated video, highlights, snapshots and heatmap (the analysis is "
                           "unchanged)")
    face.add_argument("--no-privacy", action="store_true",
                      help="do not blur faces in this run's outputs (the default)")
    p.add_argument("--camera-name", metavar="NAME",
                   help='which camera this is, e.g. "CAM03 Far side": named in every alert and spoken line')
    p.add_argument("--announce", action="store_true",
                   help="speak alerts on this computer's speaker (alarm tone + voice) when distress is found")
    p.add_argument("--start-time", metavar="HH:MM",
                   help="clock time of the video's first frame; enables time-of-day rules (active_hours) "
                        "and adds clock times to events")
    return p


def apply_overrides(cfg: dict, args) -> dict:
    """Copy command-line options into the config so every module sees one source of truth."""
    if getattr(args, "start_time", None):
        try:
            utils.parse_clock(args.start_time)
        except ValueError as exc:
            fail(f"--start-time: {exc}")
        cfg["time_of_day"]["start_time"] = args.start_time
    try:                                                         # a typo in a preset's active_hours is a clean error
        for key in utils.BEHAVIORS:
            utils.parse_hours((cfg["behaviors"].get(key) or {}).get("active_hours"))
        utils.parse_clock(cfg["time_of_day"].get("start_time"))
    except ValueError as exc:
        fail(f"time-of-day settings: {exc}")
    if args.device is not None:
        cfg["model"]["device"] = args.device
    if args.stride is not None:
        if args.stride < 1:
            fail("--stride must be 1 or more")
        cfg["video"]["frame_stride"] = int(args.stride)
    if args.max_seconds is not None:
        cfg["video"]["max_seconds"] = max(0.0, float(args.max_seconds))
    if args.no_video:
        cfg["output"]["annotated_video"] = False
        cfg["highlights"]["enabled"] = False
    if args.privacy:
        cfg["privacy"]["enabled"] = True
    if getattr(args, "no_privacy", False):
        cfg["privacy"]["enabled"] = False
    if getattr(args, "camera_name", None):
        cfg["camera"]["name"] = str(args.camera_name).strip()
    return cfg


def scenario_listing() -> str:
    """One line per preset in scenarios/: name, title and what it tracks (for --list-scenarios)."""
    names = utils.list_scenarios()
    if not names:
        return "No presets found in the scenarios/ folder."
    lines = ["Scenario presets (use with --scenario NAME):"]
    for name in names:
        try:
            cfg = utils.load_config(None, name)
        except Exception as exc:                                  # a broken preset should not hide the others
            lines.append(f"  {name:<12} (cannot be loaded: {exc})")
            continue
        classes = ", ".join(utils.class_labels(cfg))
        rules = ", ".join(utils.behavior_name(cfg, key).lower() for key in utils.BEHAVIORS
                          if cfg["behaviors"][key].get("enabled", True))
        lines.append(f"  {name:<12} {cfg['scenario']['title']}")
        lines.append(f"  {'':<12} tracks: {classes}")
        lines.append(f"  {'':<12} rules:  {rules}")
    return ascii_safe("\n".join(lines))


def things(cfg: dict, n: int | None = None) -> str:
    """Plural noun for what is tracked: 'people', 'animals', 'objects' (singular when n == 1)."""
    word = utils.entity_word(cfg).lower()
    if word == "person" and len(cfg["model"].get("classes") or [0]) > 1:    # people AND vehicles are tracked
        word = "object"
    if n == 1:
        return word
    return "people" if word == "person" else word + "s"


# --------------------------------------------------------------------------- video helpers (cv2, imported lazily)

def import_cv2():
    """Import OpenCV or stop with an install hint."""
    try:
        import cv2
        return cv2
    except ImportError:
        fail("OpenCV is not installed. Run: pip install -r requirements.txt")


def check_video_readable(path: Path) -> tuple[int, tuple[int, int]]:
    """Stop with a clear message unless the video opens and its first frame decodes.

    Uses tracker.open_video (OpenCV only, no YOLO needed). Returns (the container's frame count (0 = unknown),
    (width, height) of the real first frame).
    """
    if not path.exists():
        fail(f"video not found: {path}")
    try:
        from tracker import open_video
        cap, _fps, _size, n_frames = open_video(path)
    except ImportError:
        fail("OpenCV is not installed. Run: pip install -r requirements.txt")
    except RuntimeError as exc:                                  # open_video raises RuntimeError for bad files
        fail(str(exc))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        fail(f"cannot read video (unsupported or corrupt file): {path}")
    return n_frames, (int(frame.shape[1]), int(frame.shape[0]))     # + the real frame size (--reuse compares it)


def record_webcam(seconds: float) -> Path:
    """Record `seconds` from camera 0 to samples/webcam_<n>.mp4 and return that path.

    Frames are kept in memory as JPEG bytes with their capture times, so the file can be written with the
    REAL frame rate (webcams often deliver less than the nominal 30 fps, which would distort speeds).
    """
    if seconds <= 0:
        fail("--webcam needs a positive number of seconds")
    cv2 = import_cv2()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened() and sys.platform.startswith("win"):
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)          # Windows: DirectShow often works when the default fails
    if not cap.isOpened():
        cap.release()
        fail("cannot open the webcam (camera 0). Close other apps that use it, or use --video instead.")

    samples_dir = utils.ensure_dir(utils.ROOT / "samples")
    n = 1
    while (samples_dir / f"webcam_{n}.mp4").exists():
        n += 1
    dest = samples_dir / f"webcam_{n}.mp4"

    print(f"Recording {seconds:g} s from the webcam. Starting in 3 s - get into position ...", flush=True)
    for _ in range(3):                                      # countdown, while the camera warms up its exposure
        t_end = time.perf_counter() + 1.0
        while time.perf_counter() < t_end:
            cap.read()
    frames, stamps = [], []
    t0 = time.perf_counter()
    next_note = 5.0
    print("Recording now.", flush=True)
    try:
        while True:
            now = time.perf_counter() - t0
            if now >= seconds:
                break
            ok, frame = cap.read()
            if not ok:
                break
            good, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if good:
                frames.append(buf.tobytes())
                stamps.append(time.perf_counter() - t0)
            if now >= next_note:
                print(f"  {seconds - now:.0f} s left", flush=True)
                next_note += 5.0
    finally:
        cap.release()
    if len(frames) < 2:
        fail("the webcam delivered no frames")

    # Real frame rate = intervals between first and last frame (clamped to a sane range).
    fps = (len(stamps) - 1) / max(stamps[-1] - stamps[0], 1e-3)
    fps = float(min(60.0, max(5.0, fps)))
    first = cv2.imdecode(_to_array(frames[0]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        fail(f"cannot write {dest}")
    for buf in frames:
        writer.write(cv2.imdecode(_to_array(buf), cv2.IMREAD_COLOR))
    writer.release()
    print(f"Saved {dest}  ({len(frames)} frames, {fps:.1f} fps measured)", flush=True)
    return dest


def _to_array(jpeg_bytes: bytes):
    """bytes -> 1-D uint8 array, the form cv2.imdecode expects."""
    import numpy as np
    return np.frombuffer(jpeg_bytes, dtype=np.uint8)


# --------------------------------------------------------------------------- --reuse

def _fp32_export_source(weights) -> tuple[str, str] | None:
    """See utils.fp32_export_source (kept here for the tests that call run._fp32_export_source)."""
    return utils.fp32_export_source(weights)


def load_reusable_detections(det_path: Path, video: Path, cfg: dict, n_frames_container: int = 0,
                             orig_size=None):
    """Load <out>/detections.json if it was produced for the same video, stride, resize, detector and classes.

    Returns (tracker_result, "") on success or (None, reason) when YOLO has to run again.
    """
    from tracker import load_detections                         # light import: tracker.py loads YOLO lazily
    if not det_path.exists():
        return None, "no detections.json in the output folder yet"
    try:
        res = load_detections(det_path)
    except Exception as exc:                                    # corrupt / half-written / not a detections file
        return None, f"detections.json is unreadable ({exc})"
    needed = ("video", "fps", "stride", "orig_size", "proc_size", "duration_s", "frames_processed")
    missing = [k for k in needed if k not in res]
    if missing:
        return None, f"detections.json lacks {missing}"
    if Path(str(res["video"])).name != video.name:
        return None, f"detections are for another video ({res['video']})"
    stride = int(cfg["video"]["frame_stride"])
    if int(res["stride"]) != stride:
        return None, f"frame stride changed ({res['stride']} -> {stride})"
    width = int(cfg["video"]["resize_width"] or 0) or int(res["orig_size"][0])    # 0 = keep the original width
    if int(res["proc_size"][0]) != width:
        return None, f"resize width changed ({res['proc_size'][0]} -> {width})"
    current_sha1 = utils.weights_sha1(cfg["model"]["weights"])        # same name, new file (e.g. a retrained best.pt)
    if not utils.same_detector(res.get("model"), res.get("model_sha1"), cfg["model"]["weights"], current_sha1):
        if res.get("model") and Path(str(res["model"])).name == Path(str(cfg["model"]["weights"])).name:
            return None, f"detector file changed since these detections ({cfg['model']['weights']} was replaced)"
        return None, f"detector changed ({res['model']} -> {cfg['model']['weights']})"
    cached_classes = sorted(int(c) for c in (res.get("classes") or [0]))      # files from before presets: person only
    wanted_classes = sorted(int(c) for c in (cfg["model"].get("classes") or [0]))
    if cached_classes != wanted_classes:
        return None, f"tracked classes changed ({cached_classes} -> {wanted_classes})"
    max_s = float(cfg["video"].get("max_seconds") or 0)
    fps = float(res["fps"]) or 30.0
    if max_s > 0 and float(res["duration_s"]) > max_s + 2.0 / fps:
        return None, "cached run covers more than --max-seconds"
    if max_s > 0:
        available = (n_frames_container / fps) if n_frames_container else max_s
        if float(res["duration_s"]) < min(max_s, available) - 2.0 / fps:
            return None, "cached run covers less than --max-seconds"
    read = int(res.get("n_frames_read") or 0)
    if max_s == 0 and n_frames_container and read and read < 0.98 * n_frames_container - 2:
        return None, "cached run covers only part of the video"
    if max_s == 0 and n_frames_container and read and read > n_frames_container + 2:
        return None, "cached run is longer than the video (the video was changed or trimmed)"
    if orig_size and [int(v) for v in res["orig_size"]] != [int(v) for v in orig_size]:
        return None, f"video size changed ({res['orig_size']} -> {list(orig_size)})"
    why = utils.tracking_settings_mismatch(res, cfg)           # conf / iou / imgsz / tracker yaml (bug group C C6)
    if why:
        return None, why
    return res, ""


# --------------------------------------------------------------------------- reporting helpers

def key_settings(cfg: dict) -> dict:
    """The thresholds a reader needs to interpret the report (flat, simple values)."""
    b = cfg["behaviors"]
    enabled = [name for name in utils.BEHAVIORS if b[name].get("enabled", True)]
    nm, fa, cr = b["near_miss"], b["fall"], b["crowding"]
    return {
        "scenario": f"{cfg['scenario']['name']} ({cfg['scenario']['title']})",
        "tracked_classes": ", ".join(utils.class_labels(cfg)),
        "behaviours_enabled": ", ".join(enabled),
        "resize_width_px": cfg["video"]["resize_width"],
        "frame_stride": cfg["video"]["frame_stride"],
        "detector_conf": cfg["model"]["conf"],
        "loitering_radius_bh": b["loitering"]["radius_bh"],
        "loitering_min_duration_s": b["loitering"]["min_duration_s"],
        "zone_min_duration_s": b["zone_intrusion"]["min_duration_s"],
        "running_start_speed_bh_s": b["running"]["start_speed_bh_s"],
        "running_end_speed_bh_s": b["running"]["end_speed_bh_s"],
        "running_min_duration_s": b["running"]["min_duration_s"],
        "fall_down_min_aspect": fa["down_min_aspect"],
        "fall_min_down_s": fa["min_down_s"],
        "crowding_min_count": cr["min_count"],
        "crowding_min_duration_s": cr["min_duration_s"],
        "near_miss_distance_bh": nm["near_distance_bh"],
        "near_miss_min_rel_speed_bh_s": nm["min_rel_speed_bh_s"],
        "near_miss_ttc_s": nm["ttc_s"],
        "size_unit": utils.unit_name(cfg),
        "merge_gap_s": cfg["events"]["merge_gap_s"],
        "near_miss_ratio": cfg["events"]["near_miss_ratio"],
        "baseline_min_tracks": b["baseline"]["min_tracks"],
        "baseline_z_threshold": b["baseline"]["z_threshold"],
    }


def format_events_table(events: list[dict], width: int = 100, cfg: dict | None = None) -> str:
    """Plain-ASCII table: # | who | behaviour | severity | start-end | conf | evidence (evidence wraps).

    `who` and `behaviour` use the scenario wording stored on each event ("Car #7", "Speeding"); when an
    older event lacks them they are rebuilt from `cfg` with utils.entity_name / utils.behavior_name.
    """
    headers = ("#", "who", "behaviour", "severity", "start-end", "conf", "evidence")
    rows = []
    for e in events:
        start = e.get("start") or utils.fmt_time(e.get("start_s"))
        end = e.get("end") or utils.fmt_time(e.get("end_s"))
        who = e.get("entity_name") or utils.entity_name(cfg, e.get("entity_id"))
        what = e.get("behavior_name") or utils.behavior_name(cfg, str(e.get("behavior", "")))
        rows.append((str(e.get("event_id", "")), ascii_safe(who), ascii_safe(what), str(e.get("severity") or "-"),
                     f"{start}-{end}", f"{float(e.get('confidence', 0.0)):.2f}", ascii_safe(e.get("evidence", ""))))
    n_narrow = len(headers) - 1
    widths = [max([len(headers[i])] + [len(r[i]) for r in rows]) for i in range(n_narrow)]
    prefix = sum(widths) + 3 * n_narrow               # width taken by the narrow columns and their " | "
    text_w = max(30, width - prefix)
    sep = " | "
    lines = [sep.join(headers[i].ljust(widths[i]) for i in range(n_narrow)) + sep + headers[-1],
             "-" * min(width, prefix + text_w)]
    for r in rows:
        wrapped = textwrap.wrap(r[-1], text_w) or [""]
        lines.append(sep.join(r[i].ljust(widths[i]) for i in range(n_narrow)) + sep + wrapped[0])
        lines.extend(" " * prefix + part for part in wrapped[1:])
    return "\n".join(lines)


def format_incidents(incidents: list[dict], width: int = 100, cfg: dict | None = None) -> str:
    """Plain-ASCII list of incident chains: severity, who, title, time span and the story."""
    lines = []
    for inc in incidents:
        who = ascii_safe(inc.get("entity_name") or utils.entity_name(cfg, inc.get("entity_id")))
        head = (f"  {inc.get('chain_id', '?')}. [{str(inc.get('severity') or '-').upper()}] {who}: "
                f"{ascii_safe(inc.get('title', ''))} ({inc.get('start', '?')}-{inc.get('end', '?')})")
        lines.append(head)
        lines.extend(textwrap.wrap(ascii_safe(inc.get("story", "")), max(40, width - 7),
                                   initial_indent="       ", subsequent_indent="       "))
    return "\n".join(lines)


OUTPUT_FILES = [
    ("report.html", "evidence report - open this in a browser"),
    ("highlights.mp4", "highlight reel: only the incidents, captioned"),
    ("summary.txt", "3-6 line guard summary"),
    ("events.json", "machine-readable events and incidents"),
    ("annotated.mp4", "annotated video"),
    ("timeline.png", "who was seen when, with events"),
    ("heatmap.jpg", "where activity was seen"),
    ("snapshots", "one evidence image per event"),
    ("plots", "one speed / distance plot per event"),
    ("detections.json", "cached detections (used by --reuse)"),
]


def print_outputs(out_dir: Path, since: float = 0.0) -> None:
    """List the files written by this run (files older than `since`, a time.time() value, are leftovers)."""
    print(f"\nOutputs in {out_dir}{os.sep}")
    for name, what in OUTPUT_FILES:
        path = out_dir / name
        if not path.exists() or (path.is_file() and path.stat().st_mtime < since - 2.0):
            continue
        print(f"  {name + (os.sep if path.is_dir() else ''):<18} {what}")


# --------------------------------------------------------------------------- the pipeline

def run_tracker(video: Path, cfg: dict, det_path: Path) -> dict:
    """Detect and track objects with YOLO + ByteTrack, then cache the result for --reuse."""
    try:
        from tracker import save_detections, track_video         # lazy: ultralytics / torch load inside track_video
        result = track_video(str(video), cfg)                    # device, stride and max-seconds are already in cfg
    except ImportError as exc:                                   # ultralytics / torch / cv2 missing
        message = str(exc)
        fail(message if "pip install" in message else f"{message}. Run: pip install -r requirements.txt")
    except RuntimeError as exc:                                  # e.g. video vanished or cannot be decoded
        fail(str(exc))
    save_detections(result, det_path)
    return result


def run_optional(label: str, func, fallback):
    """Run a non-essential step (pose, rendering). On failure warn and keep going with `fallback`."""
    try:
        return func()
    except Exception as exc:                                     # never lose the events because a picture failed
        warn(f"{label} failed and was skipped: {type(exc).__name__}: {exc}"
             "  (set AQUAWATCH_DEBUG=1 to see the traceback)")
        if os.environ.get("AQUAWATCH_DEBUG"):
            import traceback
            traceback.print_exc()
        return fallback


def default_out_dir(video: Path, scenario: str | None) -> Path:
    """outputs/<video name>, with a _<scenario> suffix for every preset except campus (the default)."""
    suffix = ""
    if scenario:
        name = Path(scenario).stem                               # "traffic" or "my_presets/traffic.yaml" -> "traffic"
        if name != "campus":
            suffix = f"_{name}"
    return Path("outputs") / f"{video.stem}{suffix}"


def privacy_on(cfg: dict) -> bool:
    """Is privacy mode on? Asks privacy.is_enabled when that module is there, else reads the config."""
    try:
        import privacy
        return bool(privacy.is_enabled(cfg))
    except Exception:
        return bool((cfg.get("privacy") or {}).get("enabled"))


def main(argv=None) -> int:
    """Run the whole pipeline for one video. Returns the process exit code."""
    for stream in (sys.stdout, sys.stderr):                      # never crash on odd characters in a Windows console
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    import cpu_limits
    cpu_limits.apply()                  # below-normal priority + thread budget for this run
    if args.list_scenarios:                                      # no video needed
        print(scenario_listing())
        return 0
    if not args.video and args.webcam is None:
        parser.error("one of the arguments --video --webcam is required (or use --list-scenarios)")

    t_run = time.perf_counter()
    wall_start = time.time()
    if args.config and not Path(args.config).exists():
        fail(f"config file not found: {args.config}")
    try:
        cfg = apply_overrides(utils.load_config(args.config, args.scenario), args)
    except FileNotFoundError as exc:                             # unknown scenario name: the message lists the presets
        fail(str(exc))
    zones_path = Path(args.zones) if args.zones else None
    if zones_path is not None and not zones_path.exists():
        fail(f"zones file not found: {zones_path}  (create one with aquawatch/zone_picker.py)")

    video = record_webcam(args.webcam) if args.webcam is not None else Path(args.video)
    n_frames_container, video_size = check_video_readable(video)
    out_dir = utils.ensure_dir(args.out or default_out_dir(video, args.scenario))
    det_path = out_dir / "detections.json"
    if args.no_video:                                            # no stale video from an earlier run
        for stale in ("annotated.mp4", "highlights.mp4"):
            (out_dir / stale).unlink(missing_ok=True)
    privacy_enabled = privacy_on(cfg)
    highlights_on = bool(cfg["highlights"].get("enabled", True))

    print(f"AquaWatch analysis | video: {video} | out: {out_dir}", flush=True)
    print(f"scenario: {ascii_safe(cfg['scenario']['title'])} ({cfg['scenario']['name']}) | tracking: "
          + ", ".join(utils.class_labels(cfg))
          + (" | privacy mode: faces blurred in all outputs" if privacy_enabled else ""), flush=True)
    # steps: detect, tracks, behaviours, events, [pose], incidents, render, [highlights], report
    pool_on = bool((cfg.get("pool") or {}).get("enabled"))
    feed_on = bool(cfg["feed_health"].get("enabled"))
    steps = Steps(total=7 + (1 if args.pose else 0) + (1 if highlights_on else 0) + (1 if pool_on else 0)
                  + (1 if feed_on else 0))

    # 1. Perceive: detections with stable track IDs (or load them from the previous run).
    result, reason = None, ""
    processing_fps = None                                        # frames per second of detection + tracking
    if args.reuse:
        result, reason = load_reusable_detections(det_path, video, cfg, n_frames_container, video_size)
        if result is None:
            print(f"--reuse: cannot reuse cached detections ({reason}); running the detector.", flush=True)
    # cached pose keypoints belong to the cached track IDs: only reuse them when the detections were reused too
    detections_reused = result is not None
    if result is not None:
        with steps.step("Loading cached detections (--reuse, YOLO skipped)"):
            print(f"      {len(result['detections'])} boxes from {det_path}")
    else:
        label = f"Detecting and tracking {things(cfg)} (YOLO + ByteTrack)"
        with steps.step(label):
            result = run_tracker(video, cfg, det_path)
        track_s = result.get("tracking_s") or steps.timings.get(label) or 0.0
        if track_s > 0 and result.get("frames_processed"):
            processing_fps = round(float(result["frames_processed"]) / float(track_s), 1)
            print(f"      speed: {processing_fps:.1f} frames/s on {result.get('device')}")

    # A dropped / frozen / dark feed must never look like a quiet pool (feed_health.py, REAL_WORLD_CHECK G5).
    feed = {"ok": None, "problems": [], "summary": "feed not checked (feed_health.enabled is off)"}
    if feed_on:
        with steps.step("Checking the camera feed (complete, not frozen, not dark)"):
            import feed_health
            feed = feed_health.check(video, cfg, result)
            print("      " + ascii_safe(feed["summary"]))
            if feed.get("warnings"):
                print("      " + ascii_safe(feed_health.warnings_text(feed["warnings"])))
            utils.write_json(out_dir / "feed_health.json", feed)

    try:
        zones = utils.load_zones(zones_path, result["proc_size"])
    except (ValueError, TypeError, KeyError, ZeroDivisionError, json.JSONDecodeError) as exc:
        fail(f"zones file {zones_path} is invalid: {exc}")
    stride_dt = float(result.get("stride") or 1) / float(result.get("fps") or 30.0)
    if stride_dt > float(cfg["features"]["speed_window_s"]):
        warn(f"samples are {stride_dt:.2f} s apart (stride {result.get('stride')}), longer than "
             f"features.speed_window_s ({cfg['features']['speed_window_s']} s): speeds cannot be measured, "
             f"so running and near-miss will not fire. Use a smaller --stride.")
    if zones_path is not None and not zones:
        warn(f"{zones_path} contains no valid zone (a polygon needs 3 or more points)")
    if zones:
        print("      zones: " + ", ".join(f"{z['name']} ({len(z['points'])} points)" for z in zones))
    else:
        print("      no zones given: alerts say 'Pool' (a zones file adds Deep End / Shallow End, the wall "
              "where resting is normal, and the water outline)")

    # 2-4. Reason: tracks -> raw behaviour intervals -> merged, filtered, scored events.
    with steps.step(f"Building per-{things(cfg, 1)} tracks (smoothing, speed)"):
        # one person = one ID: join the pieces the tracker split (under water, splash, camera swing); see stitch.py.
        # detections.json keeps the raw IDs, so this runs again (with the current settings) on every --reuse.
        result, joined = stitch_tracklets(result, cfg)
        if joined:
            print(f"      joined {len(joined)} broken track piece(s) into the {things(cfg)} they belong to")
        tracks = build_tracks(result, cfg)
        tracks, on_deck = drop_deck_tracks(tracks, result, cfg)
        if on_deck:
            print(f"      left out {len(on_deck)} track(s) in the water for less than "
                  f"{0.5 * float(cfg['pool']['hold_s']):g} s (on the deck, the pool edge or a brief false detection): "
                  f"too short to ever raise an alert")
        print(f"      {len(tracks)} {things(cfg, len(tracks))} kept after filtering ghosts and tiny boxes")
    enabled = [utils.behavior_name(cfg, k).lower() for k in utils.BEHAVIORS
               if cfg["behaviors"][k].get("enabled", True)]
    with steps.step("Detecting behaviours (" + ascii_safe(", ".join(enabled)) + ")"):
        behavior_result = detect_behaviors(tracks, zones, cfg)
    with steps.step("Merging, filtering and scoring events"):
        final = build_events(behavior_result, tracks, zones, cfg)
        print(f"      {len(final['events'])} events, {len(final['near_misses'])} almost-flagged cases")
        timed = [k for k in utils.BEHAVIORS if (cfg["behaviors"].get(k) or {}).get("active_hours")]
        if final.get("ignored_outside_hours"):
            print(f"      {len(final['ignored_outside_hours'])} event(s) ignored: outside their rule's active hours")
        if timed and not cfg["time_of_day"].get("start_time"):
            warn("rules with active_hours (" + ", ".join(timed) + ") run all day: give --start-time HH:MM "
                 "(the clock time of the first frame) to apply them")

    # 4b. Pool: behaviour signals (stage 2) + temporal distress reasoning (stage 3), see pool.py.
    if pool_on:
        with steps.step("Pool: swimmer behaviour (stage 2) and distress reasoning (stage 3)"):
            def _pool():
                from pool import run_pool                     # pose model is loaded lazily inside
                return run_pool(str(video), result, tracks, zones, cfg, out_dir,
                                reuse=args.reuse and detections_reused)
            pool_res = run_optional("pool analysis", _pool, None)
            if pool_res:
                final["events"] = sorted(final["events"] + pool_res["events"],
                                         key=lambda e: (e["start_s"], e["entity_id"] if e["entity_id"] is not None else -1))
                for n, ev in enumerate(final["events"], 1):
                    ev["event_id"] = n
                final["pool"] = {"alerts": len(pool_res["stage3"]["events"]),
                                 "samples": len(pool_res["samples"]), "pose": pool_res["dashboard"]["pose"]}
                from distress import alert_kind, where_text
                for a in pool_res["stage3"]["events"]:
                    where = where_text(a).replace(" · ", ", ")             # plain ASCII for the console
                    if alert_kind(a) == "under_water":
                        print(f"      URGENT Swimmer #{a['person_id']}: possibly under water, seen again at the same "
                              f"spot at {utils.fmt_time(a['alert_time'])} ({where})")
                    elif a["event"] == "early_warning":
                        print(f"      EARLY WARNING Swimmer #{a['person_id']}: may need help at "
                              f"{utils.fmt_time(a['alert_time'])} ({where}, risk {a['risk_score']:.0%})")
                    else:
                        print(f"      ALERT Swimmer #{a['person_id']}: {a['event']} at {utils.fmt_time(a['alert_time'])} "
                              f"({where}, risk {a['risk_score']:.0%})")
                if not pool_res["stage3"]["events"]:
                    from distress import all_clear_text
                    print("      " + all_clear_text(pool_res["dashboard"]))

    # 5. Optional second opinion from body pose.
    if args.pose:
        with steps.step("Pose check (second opinion on running / loitering)"):
            def _pose():
                from pose_verify import verify_events         # lazy: pulls in ultralytics / torch
                return verify_events(str(video), result, tracks, final, cfg)
            final = run_optional("pose check", _pose, final)

    # 6. Rate every event and link one entity's events into incident chains (sets event["severity"]).
    with steps.step("Rating severity and linking events into incidents"):
        def _chains():
            from chains import build_chains                   # pure Python
            return build_chains(final["events"], cfg)
        final["incidents"] = list(run_optional("incident chains", _chains, []) or [])
        for ev in final["events"]:
            ev.setdefault("severity", None)                   # keeps the key if chains could not run
        print(f"      {len(final['incidents'])} incident chain(s)")

    # 7. Output: video, snapshots, plots, highlight reel, report.
    with steps.step("Rendering video, snapshots, plots, heatmap, timeline"):
        def _render():
            from render import render_all                     # lazy: needs cv2 + matplotlib
            return render_all(str(video), result, tracks, final, zones, cfg, out_dir)
        final = run_optional("rendering", _render, final)

    final.setdefault("highlight_reel", None)
    if highlights_on:
        with steps.step("Cutting the highlight reel (only the incidents)"):
            def _highlights():
                from highlights import make_highlight_reel   # lazy: needs cv2
                return make_highlight_reel(str(video), result, tracks, final, zones, cfg, out_dir)
            reel = run_optional("highlight reel", _highlights, None)
            final["highlight_reel"] = reel or None
            if reel:
                print(f"      wrote {reel}")

    with steps.step("Writing events.json, summary.txt, report.html"):
        meta = {
            "video": str(video),
            "duration_s": result["duration_s"],
            "fps": result["fps"],
            "orig_size": result["orig_size"],
            "proc_size": result["proc_size"],
            "frames_processed": result["frames_processed"],
            "n_tracks": len(tracks),
            "device": result.get("device"),
            "model": result.get("model"),
            "runtime_s": round(time.perf_counter() - t_run, 1),
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "stride": result["stride"],
            "zones": [z["name"] for z in zones],
            "settings": key_settings(cfg),
            "scenario": {"name": cfg["scenario"]["name"], "title": cfg["scenario"]["title"]},
            "privacy": privacy_enabled,
            "processing_fps": processing_fps,                 # None with --reuse (detection was skipped)
            "pose": bool(args.pose),
            "config": cfg,                                    # report.py reads wording and thresholds from it
        }
        meta["elapsed_s"] = meta["runtime_s"]
        meta["feed_health"] = feed
        meta["announce"] = bool(args.announce)
        announcements = []
        if pool_on or args.announce or cfg["announce"].get("enabled"):
            try:
                import announce                                   # alarm tone + voice files for the alerts
                announcements = announce.prepare(final, cfg, out_dir, speech=True)
                if announcements:
                    print(f"      {len(announcements)} spoken alert(s) written to alerts/ (voice: {announcements[0]['voice']})")
            except Exception as exc:
                warn(f"announcement audio skipped ({type(exc).__name__}: {exc})")
        summaries = [track_summary(tracks[k]) for k in sorted(tracks)]
        write_outputs(final, meta, summaries, out_dir)

    # Final console report.
    width = max(80, min(shutil.get_terminal_size((110, 20)).columns, 140))
    events = final["events"]
    print(f"\n{len(events)} event(s) in {video.name} ({utils.fmt_time(result['duration_s'])}), "
          f"{len(tracks)} {things(cfg, len(tracks))} seen:")
    print(format_events_table(events, width, cfg) if events else "No incidents flagged.")
    if feed.get("problems"):
        print(ascii_safe(feed["summary"]) + " - results cover only the usable part of the video.")
    if final.get("incidents"):
        print(f"\n{len(final['incidents'])} incident chain(s) (events of one entity linked over time):")
        print(format_incidents(final["incidents"], width, cfg))
    if final.get("near_misses"):
        print(f"\n{len(final['near_misses'])} almost-flagged case(s) stayed below the rules; "
              "they are listed in the report.")
    unusual = final.get("unusual_tracks") or []
    if unusual:
        print("Unusual compared with the rest of the scene: " +
              ", ".join(ascii_safe(utils.entity_name(cfg, u["entity_id"])) for u in unusual))
    summary_file = out_dir / "summary.txt"
    if summary_file.exists():
        print("\nGuard summary:\n" + ascii_safe(summary_file.read_text(encoding="utf-8")).rstrip())
    print_outputs(out_dir, wall_start)
    if announcements and (args.announce or cfg["announce"].get("enabled")):
        import announce
        print("\nSpeaker announcement:")
        announce.speak_all(announcements, cfg, out_dir)
    print(f"\nTotal time {time.perf_counter() - t_run:.1f} s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
