"""Optional second opinion (DESIGN 3.5): does the body posture agree with the flagged behaviour?

Only used with `run.py --pose`. For each *running* and *loitering* event we sample a few frames,
crop the person, run YOLO11n-pose and measure two simple posture numbers:

  lean_deg        angle of the hip-middle -> shoulder-middle line from vertical (0 = upright)
  leg_spread_bh   distance between the two ankles, in body-heights

  running   agrees if the legs ever stretch wide (a stride) OR the torso leans forward
  loitering agrees if the person is mostly upright (standing, not crouching / lying)

The check can only add information: every failure path ends in verified = "uncertain" with a
note, and nothing in here is allowed to crash the main run.

ultralytics is imported inside a function only; cv2 is needed for reading frames.
"""
from __future__ import annotations

import math
import sys

import numpy as np

from tracker import resize_frame, resolve_device, resolve_weights, to_numpy

POSE_BEHAVIORS = ("running", "loitering")      # zone_intrusion has no posture signature
PAD_FRACTION = 0.20                            # crop = person box grown by 20 % each side (DESIGN 3.5)
MIN_CROP_PX = 8                                # ignore degenerate crops

# COCO keypoint indices used by the check.
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_ANKLE, R_ANKLE = 15, 16
REQUIRED_KEYPOINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_ANKLE, R_ANKLE)


# --------------------------------------------------------------------------- pure helpers (NumPy only)

def sample_indices(n: int, k: int) -> list[int]:
    """Up to k indices spread evenly over range(n) (always includes first and last when n > k)."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, k)})


def padded_crop_box(box, frame_w: int, frame_h: int, pad: float = PAD_FRACTION) -> tuple[int, int, int, int]:
    """Person box grown by `pad` (fraction of its width / height) and clipped to the frame."""
    x1, y1, x2, y2 = (float(v) for v in box)
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    return (max(0, int(math.floor(x1 - pw))), max(0, int(math.floor(y1 - ph))),
            min(int(frame_w), int(math.ceil(x2 + pw))), min(int(frame_h), int(math.ceil(y2 + ph))))


def pose_metrics(kp_xy, kp_conf, box_h: float, min_conf: float) -> dict | None:
    """Lean and leg spread of one person, or None if a needed keypoint is not visible enough.

    kp_xy   (17, 2) COCO keypoints in pixels
    kp_conf (17,)   keypoint confidences
    box_h   the person's box height in the same pixels (the body-height unit)
    """
    kp_xy = np.asarray(kp_xy, dtype=float)
    kp_conf = np.asarray(kp_conf, dtype=float)
    if kp_xy.shape[0] < 17 or kp_conf.shape[0] < 17 or box_h <= 0:
        return None
    needed = list(REQUIRED_KEYPOINTS)
    if not np.all(kp_conf[needed] >= min_conf) or not np.all(np.isfinite(kp_xy[needed])):
        return None

    shoulder_mid = kp_xy[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
    hip_mid = kp_xy[[L_HIP, R_HIP]].mean(axis=0)
    vx, vy = shoulder_mid - hip_mid                     # image y grows downwards, so "up" is -y
    if math.hypot(vx, vy) < 1e-6:
        return None
    lean_deg = math.degrees(math.atan2(abs(vx), -vy))   # 0 = upright, 90 = horizontal torso
    ankle_gap = float(np.hypot(*(kp_xy[L_ANKLE] - kp_xy[R_ANKLE])))
    return {"lean_deg": float(lean_deg), "leg_spread_bh": ankle_gap / float(box_h)}


def judge_event(behavior: str, frame_metrics: list[dict], frames_checked: int, cfg: dict) -> tuple:
    """Combine per-frame posture numbers into (verified, pose_dict) for one event.

    verified is True / False / "uncertain" (too few usable frames).
    """
    pcfg = cfg["pose"]
    usable = len(frame_metrics)
    pose = {"frames_checked": int(frames_checked), "usable_frames": int(usable),
            "median_lean_deg": None, "max_leg_spread_bh": None, "agrees": None, "note": ""}
    if usable:
        leans = [m["lean_deg"] for m in frame_metrics]
        spreads = [m["leg_spread_bh"] for m in frame_metrics]
        median_lean, max_spread = float(np.median(leans)), float(max(spreads))
        pose["median_lean_deg"] = round(median_lean, 1)
        pose["max_leg_spread_bh"] = round(max_spread, 2)

    needed = max(1, int(pcfg["min_pose_frames"]))      # at least one frame, or there is nothing to judge
    if usable < needed:
        pose["note"] = (f"Uncertain: only {usable} of {frames_checked} checked frames showed shoulders, hips "
                        f"and ankles clearly (need >= {needed}).")
        return "uncertain", pose

    if behavior == "running":
        spread_ok = max_spread >= pcfg["run_leg_spread_bh"]
        lean_ok = median_lean >= pcfg["run_lean_deg"]
        agrees = bool(spread_ok or lean_ok)
        pose["note"] = (f"Pose {'supports' if agrees else 'does not support'} running: widest ankle gap "
                        f"{max_spread:.2f} body-heights (stride if >= {pcfg['run_leg_spread_bh']:g}), median torso "
                        f"lean {median_lean:.1f} deg (running if >= {pcfg['run_lean_deg']:g}); "
                        f"{usable} of {frames_checked} frames usable.")
    else:                                               # loitering: a standing person is upright
        agrees = bool(median_lean <= pcfg["upright_max_lean_deg"])
        pose["note"] = (f"Pose {'supports' if agrees else 'does not support'} standing in place: median torso "
                        f"lean {median_lean:.1f} deg (upright if <= {pcfg['upright_max_lean_deg']:g}); "
                        f"{usable} of {frames_checked} frames usable.")
    pose["agrees"] = agrees
    return agrees, pose


def _best_person(result, target_box_in_crop, min_conf: float) -> dict | None:
    """Posture numbers of the right person in one pose result, or None.

    DESIGN says "take the most confident person". When several people are in the padded crop we
    first prefer detections whose centre lies inside the tracked person's own box, so a
    neighbour who happens to score higher is not measured by mistake.
    """
    boxes, kps = getattr(result, "boxes", None), getattr(result, "keypoints", None)
    if boxes is None or kps is None or len(boxes) == 0 or kps.xy is None or kps.conf is None:
        return None
    det_conf = to_numpy(boxes.conf).reshape(-1)
    det_xyxy = to_numpy(boxes.xyxy).reshape(-1, 4)
    kp_xy, kp_conf = to_numpy(kps.xy), to_numpy(kps.conf)

    tx1, ty1, tx2, ty2 = target_box_in_crop
    cx, cy = (det_xyxy[:, 0] + det_xyxy[:, 2]) / 2, (det_xyxy[:, 1] + det_xyxy[:, 3]) / 2
    inside = (cx >= tx1) & (cx <= tx2) & (cy >= ty1) & (cy <= ty2)
    pool = np.where(inside)[0] if inside.any() else np.arange(len(det_conf))
    best = int(pool[np.argmax(det_conf[pool])])
    return pose_metrics(kp_xy[best], kp_conf[best], ty2 - ty1, min_conf)


# --------------------------------------------------------------------------- video + model plumbing

def _read_frame(cap, frame_idx: int, proc_size):
    """Seek to an original frame number and return it at processing size (None on failure)."""
    import cv2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return resize_frame(frame, proc_size)


def _set_uncertain(event: dict, note: str, frames_checked: int = 0) -> None:
    """Mark an event as not verifiable, with the reason."""
    event["verified"] = "uncertain"
    event["pose"] = {"frames_checked": int(frames_checked), "usable_frames": 0, "median_lean_deg": None,
                     "max_leg_spread_bh": None, "agrees": None, "note": note}


def _verify_one(cap, pose_model, device, event: dict, track, proc_size, cfg: dict) -> None:
    """Run the pose check for a single event (raises on unexpected errors; caller catches)."""
    pcfg, mcfg = cfg["pose"], cfg["model"]
    if track is None:
        raise LookupError(f"track #{event.get('entity_id')} not found")

    t = np.asarray(track.t, dtype=float)
    inside = np.where((t >= event["start_s"] - 1e-6) & (t <= event["end_s"] + 1e-6))[0]
    chosen = [inside[k] for k in sample_indices(len(inside), int(pcfg["max_frames_per_event"]))]

    frame_w, frame_h = int(proc_size[0]), int(proc_size[1])
    frame_metrics, checked = [], 0
    for i in chosen:
        frame = _read_frame(cap, track.frame[i], proc_size)
        if frame is None:
            continue
        box = [float(v) for v in track.box[i]]
        x1, y1, x2, y2 = padded_crop_box(box, frame_w, frame_h)
        if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
            continue
        crop = frame[y1:y2, x1:x2]
        checked += 1
        results = pose_model.predict(crop, imgsz=mcfg["imgsz"], conf=mcfg["conf"], device=device, verbose=False)
        if not results:
            continue
        target = (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1)      # tracked box in crop pixels
        metrics = _best_person(results[0], target, pcfg["min_keypoint_conf"])
        if metrics is not None:
            frame_metrics.append(metrics)

    if checked == 0:
        _set_uncertain(event, "Uncertain: no frame of this event could be read for the pose check.")
        return
    verified, pose = judge_event(event["behavior"], frame_metrics, checked, cfg)
    event["verified"], event["pose"] = verified, pose


# --------------------------------------------------------------------------- public entry point

def verify_events(video_path, tracker_result: dict, tracks: dict, final: dict, cfg: dict) -> dict:
    """Add a pose-based verdict to every running / loitering event in `final` (DESIGN 3.5).

    Sets event['verified'] to True / False / 'uncertain' and event['pose'] to a dict of the
    measured numbers plus a plain-English note. Never raises: any problem becomes 'uncertain'.
    Returns `final` (modified in place).
    """
    try:
        events = [e for e in final.get("events", []) if e.get("behavior") in POSE_BEHAVIORS]
    except Exception:                                   # final is not the expected shape: nothing to do
        return final
    if not events:
        return final

    try:
        return _verify_events(video_path, tracker_result, tracks, final, events, cfg)
    except Exception as exc:                            # last safety net
        for e in events:
            if e.get("verified") is None:
                _set_uncertain(e, f"Uncertain: pose check failed ({type(exc).__name__}: {exc}).")
        return final


def _verify_events(video_path, tracker_result, tracks, final, events, cfg) -> dict:
    """Load the pose model and video once, then check every eligible event."""
    cap = pose_model = None
    try:
        import cv2
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is not installed (pip install ultralytics)") from exc
        import cpu_limits
        cpu_limits.apply()              # keep torch / Ultralytics on the laptop's thread budget
        device = resolve_device(tracker_result.get("device") or cfg["model"].get("device", "auto"))
        pose_model = YOLO(resolve_weights(cfg["model"]["pose_weights"]))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video '{video_path}'")
    except Exception as exc:                            # model / video unavailable: every event is uncertain
        if cap is not None:
            cap.release()
        for e in events:
            _set_uncertain(e, f"Uncertain: pose check unavailable ({type(exc).__name__}: {exc}).")
        print(f"WARNING: pose check skipped: {exc}", file=sys.stderr)
        return final

    proc_size = tracker_result["proc_size"]
    try:
        for n, event in enumerate(events, 1):
            try:
                entity = event.get("entity_id")
                track = tracks.get(entity)
                if track is None:
                    try:
                        track = tracks.get(int(entity))
                    except (TypeError, ValueError):
                        track = None
                _verify_one(cap, pose_model, device, event, track, proc_size, cfg)
            except Exception as exc:                    # one bad event must not stop the others
                _set_uncertain(event, f"Uncertain: pose check failed ({type(exc).__name__}: {exc}).")
            print(f"  pose check {n}/{len(events)}: event {event.get('event_id')} "
                  f"({event.get('behavior')}, person #{event.get('entity_id')}) -> {event.get('verified')}",
                  flush=True)
    finally:
        cap.release()
    return final
