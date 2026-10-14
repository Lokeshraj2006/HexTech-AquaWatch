"""Perception stage: YOLO11n detection + ByteTrack tracking (DESIGN 3.1 and 4.5).

Reads a video, runs a pretrained YOLO detector on every `frame_stride`-th frame and lets
Ultralytics' built-in ByteTrack give every object a stable track ID. Which COCO classes are
tracked comes from `model.classes` (default [0] = person; scenario presets add vehicles or
animals). The output is a plain dict (TrackerResult) that is also saved as detections.json,
so the later stages (features, behaviours, events, ...) never need the video or the neural
network again.

torch / ultralytics / cv2 are imported inside the functions only, so `run.py --reuse`
(which just calls load_detections) works on a machine without the vision stack.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

import utils

FALLBACK_FPS = 30.0       # used when the video file reports fps = 0 / NaN
PROGRESS_STEP_PCT = 5     # print a progress line every ~5 % of the video


# --------------------------------------------------------------------------- small helpers

def resolve_device(name) -> str:
    """Turn the config / CLI device name into something Ultralytics accepts.

    'auto' -> '0' (first GPU) if CUDA is available, else 'cpu'. A GPU that is asked for but
    not available falls back to 'cpu' with a warning instead of crashing.
    """
    name = "auto" if name is None else str(name).strip().lower()
    if name in ("", "auto"):
        try:
            import torch
            return "0" if torch.cuda.is_available() else "cpu"
        except Exception:                      # torch missing or broken -> CPU
            return "cpu"
    if name == "cpu" or name == "mps":
        return name
    if name.isdigit() or name.startswith("cuda"):      # '0', '1', 'cuda', 'cuda:0'
        try:
            import torch
            if torch.cuda.is_available():
                return name
        except Exception:
            pass
        print(f"WARNING: device '{name}' requested but CUDA is not available; using cpu.", file=sys.stderr)
        return "cpu"
    return name                                         # e.g. '0,1' - let Ultralytics decide


def open_video(path):
    """Open a video. Returns (cap, fps, (W, H), frame_count).

    fps falls back to 30.0 (with a warning) if the file reports 0 / NaN. frame_count is 0
    when the container does not know it. Raises RuntimeError with a readable message if the
    file is missing or cannot be decoded.
    """
    import cv2

    path = str(path)
    if not Path(path).exists():
        raise RuntimeError(f"Video file not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video '{path}' (unsupported codec or corrupt file?)")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not math.isfinite(fps) or fps <= 0:
        print(f"WARNING: video reports fps={fps}; assuming {FALLBACK_FPS:g} fps "
              "(timestamps will only be approximate).", file=sys.stderr)
        fps = FALLBACK_FPS

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    n_frames = int(n_frames) if n_frames is not None and math.isfinite(n_frames) and n_frames > 0 else 0

    if width <= 0 or height <= 0:                       # metadata missing: measure on a real frame
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Video '{path}' opened but no frame could be read.")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, float(fps), (width, height), n_frames


def proc_size_for(orig_size, resize_width) -> tuple[int, int]:
    """Processing frame size (W, H): resized to `resize_width` wide, aspect ratio kept.

    resize_width of 0 / None keeps the original size.
    """
    w0, h0 = int(orig_size[0]), int(orig_size[1])
    if not resize_width or int(resize_width) <= 0 or w0 <= 0:
        return w0, h0
    w = int(resize_width)
    h = max(1, int(round(h0 * w / w0)))
    return w, h


def resize_frame(frame, proc_size):
    """Resize a BGR frame to the processing size (INTER_AREA when shrinking)."""
    import cv2

    h, w = frame.shape[:2]
    pw, ph = int(proc_size[0]), int(proc_size[1])
    if (w, h) == (pw, ph):
        return frame
    interp = cv2.INTER_AREA if pw < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (pw, ph), interpolation=interp)


def to_numpy(x) -> np.ndarray:
    """Torch tensor (any device) or array-like -> NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def resolve_weights(name) -> str:
    """Find a weights file: absolute path, then project root, then as given.

    If nothing exists locally the bare name is returned and Ultralytics downloads the
    official weights (e.g. 'yolo11n.pt') automatically on first use.
    """
    p = Path(str(name))
    if p.is_absolute():
        return str(p)
    if (utils.ROOT / p).exists():
        return str(utils.ROOT / p)
    return str(name)


def check_model_present(resolved: str, name) -> None:
    """A model inside a folder (an OpenVINO export or a trained model) is never downloaded by Ultralytics: say how to
    get it instead of Ultralytics' long FileNotFoundError. A bare name such as 'yolo11n.pt' is left to Ultralytics."""
    p = Path(str(name))
    if p.parent != Path(".") and not Path(resolved).exists():
        raise RuntimeError(f"Model files not found: {p.as_posix()}. Run once: python aquawatch/get_models.py")


def _resolve_tracker_yaml(name) -> str:
    """Absolute path of the ByteTrack yaml (relative names: the code folder, the project root, cwd)."""
    p = Path(str(name))
    if p.is_absolute():
        return str(p)
    for base in (utils.CODE_DIR, utils.ROOT, Path.cwd()):
        if (base / p).exists():
            return str((base / p).resolve())
    return str(name)             # maybe a built-in name such as 'bytetrack.yaml'


def tracker_settings(name) -> dict:
    """The ByteTrack yaml: {"path", "low", "sha1"}. low = track_low_thresh (None if unreadable); sha1 of the file
    ('' for a built-in name), so run.py --reuse can tell a changed tracker yaml apart."""
    import hashlib
    path = _resolve_tracker_yaml(name)
    low, sha1 = None, ""
    try:
        data = Path(path).read_bytes()
        sha1 = hashlib.sha1(data).hexdigest()
        import yaml
        low = float((yaml.safe_load(data) or {}).get("track_low_thresh"))
    except (OSError, TypeError, ValueError):
        pass
    return {"path": path, "low": low, "sha1": sha1}


def _tracked_rows(result, frame_idx: int, t: float) -> list[list]:
    """Pull [frame_idx, t, id, x1, y1, x2, y2, conf, cls] rows out of one Ultralytics Results object.

    The 9th element is the COCO class id (0 person, 2 car, ...). Boxes without a track id
    (ByteTrack has not confirmed them yet) are skipped.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:      # no detections, or no confirmed tracks
        return []
    ids = to_numpy(boxes.id).reshape(-1).astype(int)
    xyxy = to_numpy(boxes.xyxy).reshape(-1, 4).astype(float)
    conf = to_numpy(boxes.conf).reshape(-1).astype(float)
    cls_raw = getattr(boxes, "cls", None)
    classes = to_numpy(cls_raw).reshape(-1).astype(int) if cls_raw is not None else np.zeros(len(ids), dtype=int)
    rows = []
    for tid, (x1, y1, x2, y2), c, k in zip(ids, xyxy, conf, classes):
        rows.append([int(frame_idx), round(float(t), 4), int(tid),
                     round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2),
                     round(float(c), 3), int(k)])
    return rows


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# --------------------------------------------------------------------------- main entry point

def track_video(video_path, cfg: dict, device_override=None, max_seconds=None, stride=None,
                progress: bool = True) -> dict:
    """Detect and track objects in a video; returns a TrackerResult dict (DESIGN 3.1).

    video_path      path of the video file
    cfg             config dict from utils.load_config()
    device_override 'auto' | 'cpu' | '0' ...; None uses cfg['model']['device']
    max_seconds     only the first N seconds (None -> cfg['video']['max_seconds']; 0 = whole video)
    stride          process every Nth frame (None -> cfg['video']['frame_stride'])
    progress        print a progress line about every 5 %

    Frames are resized to cfg['video']['resize_width'] before detection, so every box in the
    result is in processing pixels. Frame numbers and times refer to the ORIGINAL video.
    Only the COCO classes in cfg['model']['classes'] are tracked (default [0] = person).
    """
    vcfg, mcfg = cfg["video"], cfg["model"]
    classes = [int(c) for c in (mcfg.get("classes") or [0])]
    stride = max(1, int(stride if stride is not None else vcfg["frame_stride"]))
    if max_seconds is None:
        max_seconds = vcfg.get("max_seconds", 0)
    device = resolve_device(device_override if device_override is not None else mcfg.get("device", "auto"))

    cap, fps, orig_size, frame_count = open_video(video_path)
    try:
        proc_size = proc_size_for(orig_size, vcfg.get("resize_width"))
        max_frames = max(1, int(round(float(max_seconds) * fps))) if max_seconds and max_seconds > 0 else None
        total = frame_count if frame_count > 0 else 0       # frames we expect to read (progress only)
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames

        if progress:
            print(f"Tracking {video_path}: {orig_size[0]}x{orig_size[1]} @ {fps:g} fps -> "
                  f"processing {proc_size[0]}x{proc_size[1]}, frame stride {stride}, device {device}", flush=True)

        # Heavy imports only here, so the rest of the project works without them.
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is not installed. Install it with: pip install ultralytics "
                              "(or run with --reuse to use saved detections).") from exc
        import cpu_limits
        cpu_limits.apply()              # after the import: Ultralytics would otherwise run torch on cpu_count - 1 threads
        weights = resolve_weights(mcfg["weights"])
        check_model_present(weights, mcfg["weights"])
        if Path(weights).suffix.lower() in (".xml", ".onnx") or Path(weights).is_dir():
            # an exported model (OpenVINO / ONNX, models/quant): its runtime would use every core; cap it the same way,
            # and give torch 1 thread (it only does pre- and post-processing around the exported model)
            import quantize
            quantize.cap_runtime_threads(threads=cpu_limits.threads(), torch_threads=1)
        model = YOLO(weights)
        tsettings = tracker_settings(mcfg["tracker"])
        tracker_yaml = tsettings["path"]
        # Bug hunt 2026-10-11 (tracking high): with conf = the yaml's HIGH value, ByteTrack's second matching step
        # (boxes between track_low_thresh and track_high_thresh) never runs. Passing conf = track_low_thresh (step
        # C5 of PREREGISTRATION_bug_group_C.md) FAILED its check on fresh runs (docs/RESULTS.md section Z), so the
        # detector keeps model.conf; the low threshold is recorded for a later, better-tested attempt.
        det_conf = float(mcfg["conf"])
        # A swimmer state detector's classes are states of ONE body (horizontal / vertical / ...): overlapping boxes
        # of different states are the same swimmer, so NMS must compare across classes, or one swimmer gets two
        # boxes and ByteTrack gives the second one a new ID. COCO presets keep the default per-class NMS.
        agnostic = bool(utils.state_classes(cfg))

        detections: list[list] = []
        frame_idx = 0           # index of the next source frame
        processed = 0
        next_report = PROGRESS_STEP_PCT
        started = time.perf_counter()

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % stride != 0:                 # skipped frame: decode cheaply, no resize / YOLO
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if processed == 0:                          # trust the real frame over the metadata
                real = (frame.shape[1], frame.shape[0])
                if real != tuple(orig_size):
                    print(f"WARNING: frame size {real} differs from metadata {tuple(orig_size)}; using the real size.",
                          file=sys.stderr)
                    orig_size = real
                    proc_size = proc_size_for(orig_size, vcfg.get("resize_width"))

            small = resize_frame(frame, proc_size)
            # persist=True keeps ByteTrack's state between calls, which is what gives stable IDs.
            results = model.track(small, persist=True, classes=classes, conf=det_conf, iou=mcfg["iou"],
                                  imgsz=mcfg["imgsz"], tracker=tracker_yaml, device=device, verbose=False,
                                  agnostic_nms=agnostic)
            if results:
                detections.extend(_tracked_rows(results[0], frame_idx, frame_idx / fps))
            frame_idx += 1
            processed += 1

            if progress:
                if total > 0:
                    pct = 100.0 * frame_idx / total
                    if pct >= next_report:
                        elapsed = time.perf_counter() - started
                        eta = elapsed * (total - frame_idx) / max(frame_idx, 1)
                        print(f"  tracking {min(pct, 100.0):3.0f}%  (frame {frame_idx}/{total}, "
                              f"{processed / max(elapsed, 1e-9):.1f} processed fps, ETA {_fmt_eta(eta)})", flush=True)
                        while next_report <= pct:
                            next_report += PROGRESS_STEP_PCT
                elif processed % 100 == 0:               # unknown length: report by count
                    print(f"  tracking ... {processed} frames processed", flush=True)
    finally:
        cap.release()

    n_frames_read = frame_idx
    if n_frames_read == 0:
        raise RuntimeError(f"Video '{video_path}' opened but no frame could be read.")

    took = time.perf_counter() - started          # detection + tracking time, without loading the model
    result = {
        "video": str(video_path),
        "fps": float(fps),
        "stride": int(stride),
        "orig_size": [int(orig_size[0]), int(orig_size[1])],
        "proc_size": [int(proc_size[0]), int(proc_size[1])],
        "n_frames_read": int(n_frames_read),
        "duration_s": float(n_frames_read / fps),
        "frames_processed": int(processed),
        "device": device,
        "model": Path(str(mcfg["weights"])).name,
        "model_sha1": utils.weights_sha1(mcfg["weights"]),   # tells a retrained best.pt apart (run.py --reuse)
        "classes": classes,                       # which COCO classes were tracked (checked by run.py --reuse)
        # the settings that decide the boxes and IDs (checked by run.py --reuse since 2026-10-12, bug group C step C6)
        "conf": float(mcfg["conf"]), "det_conf": float(det_conf), "iou": float(mcfg["iou"]),
        "imgsz": int(mcfg["imgsz"]), "tracker": str(mcfg["tracker"]), "tracker_sha1": tsettings["sha1"],
        "tracking_s": round(float(took), 3),      # run.py shows frames_processed / tracking_s as the speed
        "detections": detections,
    }
    if progress:
        n_ids = len({row[2] for row in detections})
        print(f"Tracking done: {processed} frames in {took:.1f} s, "
              f"{len(detections)} boxes, {n_ids} track IDs.", flush=True)
    return result


# --------------------------------------------------------------------------- save / load

def save_detections(result: dict, path) -> None:
    """Write a TrackerResult to detections.json."""
    utils.write_json(path, result)


def load_detections(path) -> dict:
    """Read a TrackerResult back from detections.json (raises ValueError if it is not one)."""
    data = utils.read_json(path)
    if not isinstance(data, dict) or "detections" not in data or "proc_size" not in data:
        raise ValueError(f"{path} is not a detections file written by tracker.py")
    return data


# --------------------------------------------------------------------------- CLI (debug use)

def _main(argv=None) -> int:
    """`python tracker.py --video X` runs only the tracking stage and writes detections.json."""
    import argparse

    ap = argparse.ArgumentParser(description="Run only the YOLO + ByteTrack stage and save detections.json.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scenario", default=None, help="preset name from scenarios/ (sets which classes are tracked)")
    ap.add_argument("--out", default="detections.json")
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    args = ap.parse_args(argv)
    try:
        result = track_video(args.video, utils.load_config(args.config, args.scenario),
                             device_override=args.device, max_seconds=args.max_seconds, stride=args.stride)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    save_detections(result, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
