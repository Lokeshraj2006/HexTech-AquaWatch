"""Camera feed health: was the video complete and usable? (REAL_WORLD_CHECK G5)

A dropped camera feed must never look like a quiet pool. On every run this module reads the video once more (cheap: a
small grey copy of about feed_health.sample_hz frames per second) and flags:
  * ended_early  the file has fewer frames than its own header says (recording cut off / file damaged);
  * gap          the frame timestamps jump (part of the recording is missing);
  * frozen       the picture does not change at all for problem_s seconds (a stuck camera or encoder);
  * dark         the picture is black or blank (no light, lens covered, no signal) for problem_s seconds.
site_check() adds, for a multi-camera site, a camera whose recording ends long before the others.
Two VIEW warnings (results less reliable, never a change to an alert; REAL_WORLD_CHECK G7):
  * moving       the whole picture shifts (handheld phone, drone, camera turning): movement signals unreliable;
  * top_down     people look the same size at the top and bottom of the picture (camera looking straight down):
                 posture read from box shape assumes a side view (view_check, from the detections).

    feed_health.measure(video_path, cfg)              -> {"fps", "frames_header", "frames_read", "samples", ...}
    feed_health.assess(measured, cfg)                 -> {"ok", "problems": [...], "summary"}
    feed_health.check(video_path, cfg)                -> measure + assess (never raises: a failed check says so)
    feed_health.site_check(cameras, cfg)              -> {camera_id: [problem, ...]}

Only measure() uses OpenCV (imported inside it); the decisions are plain NumPy so they can be tested on their own.
Thresholds: config.yaml section `feed_health`.
"""
from __future__ import annotations

import numpy as np

import utils

KIND_TEXT = {"ended_early": "recording ended early", "gap": "part of the recording is missing",
             "frozen": "picture frozen", "dark": "picture dark or blank"}


def _settings(cfg: dict | None) -> dict:
    """The feed_health settings with defaults filled in."""
    return {**utils.DEFAULT_CONFIG["feed_health"], **((cfg or {}).get("feed_health") or {})}


def measure(video_path, cfg: dict | None) -> dict:
    """Read the whole video once: frame count, timestamp gaps, and per sample (about sample_hz per second) the mean
    brightness, the contrast (std) and the change since the previous sample, on a 64 px wide grey copy."""
    import cv2
    st = _settings(cfg)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        header = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        header = int(header) if header and np.isfinite(header) and header > 0 else 0
        step = max(1, int(round(fps / float(st["sample_hz"]))))
        samples, gaps, prev, prev_ms, n, window = [], [], None, None, 0, None
        while cap.grab():
            ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            if prev_ms is not None and ms - prev_ms > 1000.0 * float(st["gap_s"]):
                gaps.append([round(prev_ms / 1000.0, 2), round(ms / 1000.0, 2)])
            prev_ms = ms
            if n % step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    small = cv2.resize(frame, (64, max(1, int(64 * frame.shape[0] / max(1, frame.shape[1])))),
                                       interpolation=cv2.INTER_AREA)
                    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) if small.ndim == 3 else \
                        small.astype(np.float32)
                    change = float(np.mean(np.abs(grey - prev))) if prev is not None else None
                    shift = None                  # whole-picture shift since the last sample, share of the width
                    if prev is not None:
                        if window is None:
                            window = cv2.createHanningWindow((grey.shape[1], grey.shape[0]), cv2.CV_32F)
                        (dx, dy), _ = cv2.phaseCorrelate(prev, grey, window)
                        shift = round(float(np.hypot(dx, dy)) / grey.shape[1], 4)
                    samples.append([round(n / fps, 3), round(float(grey.mean()), 2), round(float(grey.std()), 2),
                                    None if change is None else round(change, 3), shift])
                    prev = grey
            n += 1
    finally:
        cap.release()
    return {"fps": fps, "frames_header": header, "frames_read": n, "samples": samples, "gaps": gaps}


def _runs(times: list[float], flags: list[bool], min_s: float, dt: float) -> list[tuple[float, float]]:
    """(start, end) of every run of consecutive True flags lasting at least min_s (a sample covers dt seconds)."""
    out, start = [], None
    for t, f in zip(list(times) + [None], list(flags) + [False]):
        if f and start is None:
            start = t
        elif not f and start is not None:
            end = (t if t is not None else times[-1] + dt)
            if end - start >= min_s - 1e-9:
                out.append((round(start, 2), round(end, 2)))
            start = None
    return out


def assess(measured: dict, cfg: dict | None) -> dict:
    """Decide from measure()'s numbers. Returns {"ok", "problems": [{"kind", "start_s", "end_s", "text"}], "summary",
    "frames_header", "frames_read"}."""
    st = _settings(cfg)
    fps = float(measured.get("fps") or 30.0)
    header, read = int(measured.get("frames_header") or 0), int(measured.get("frames_read") or 0)
    problems = []
    if header and (header - read) / fps >= float(st["end_missing_s"]):
        problems.append({"kind": "ended_early", "start_s": round(read / fps, 2), "end_s": round(header / fps, 2),
                         "text": f"recording ended early: {read} of {header} frames could be read "
                                 f"({(header - read) / fps:.1f} s missing at the end)"})
    for a, b in measured.get("gaps") or []:
        problems.append({"kind": "gap", "start_s": a, "end_s": b,
                         "text": f"part of the recording is missing: {b - a:.1f} s jump at {utils.fmt_time(a)}"})
    samples = measured.get("samples") or []
    if samples:
        times = [s[0] for s in samples]
        dt = float(st["sample_hz"]) and 1.0 / float(st["sample_hz"])
        dark = [s[1] <= float(st["dark_mean"]) or s[2] <= float(st["blank_std"]) for s in samples]
        frozen = [s[3] is not None and s[3] <= float(st["frozen_change"]) for s in samples]
        for kind, flags in (("dark", dark), ("frozen", frozen)):
            for a, b in _runs(times, flags, float(st["problem_s"]), dt):
                if kind == "frozen":        # the frozen picture began one sample earlier (the first unchanged one)
                    a = round(max(0.0, a - dt), 2)
                problems.append({"kind": kind, "start_s": a, "end_s": b,
                                 "text": f"{KIND_TEXT[kind]} from {utils.fmt_time(a)} to {utils.fmt_time(b)} "
                                         f"({b - a:.0f} s)"})
    problems.sort(key=lambda p: p["start_s"])
    warnings = []
    if samples and st["view_warnings"]:              # OFF by default: the G7 pre-registered check failed
        moving = [len(s) > 4 and s[4] is not None and s[4] >= float(st["move_frac"]) for s in samples]
        for a, b in _runs(times, moving, float(st["problem_s"]), dt):
            a = round(max(0.0, a - dt), 2)            # it was already moving between the previous sample and this one
            warnings.append({"kind": "moving", "start_s": a, "end_s": b,
                             "text": f"camera moving from {utils.fmt_time(a)} to {utils.fmt_time(b)} ({b - a:.0f} s): "
                                     f"movement-based signals are unreliable there"})
    return {"ok": not problems, "problems": problems, "summary": summary_text(problems), "warnings": warnings,
            "frames_header": header, "frames_read": read}


def view_check(tracker_result: dict, cfg: dict | None) -> list[dict]:
    """Top-down view? In a side view far people (top of the picture) look smaller than near ones (bottom). Compares
    the median box height in the bottom third with the top third; [] when it looks like a side view or when there
    are too few boxes in either third to tell."""
    st = _settings(cfg)
    height = float((tracker_result.get("proc_size") or (0, 0))[1] or 0)
    rows = [r for r in tracker_result.get("detections") or [] if len(r) >= 7 and float(r[6]) > float(r[4])]
    if not height or not rows:
        return []
    mid = np.array([(float(r[4]) + float(r[6])) / 2 for r in rows])
    tall = np.array([float(r[6]) - float(r[4]) for r in rows])
    top, bottom = tall[mid < height / 3], tall[mid >= 2 * height / 3]
    need = int(st["view_min_boxes"])
    if len(top) < need or len(bottom) < need:
        return []
    ratio = float(np.median(bottom) / max(np.median(top), 1e-6))
    if ratio >= float(st["top_down_ratio"]):
        return []
    return [{"kind": "top_down", "start_s": None, "end_s": None, "ratio": round(ratio, 2),
             "text": f"people look about the same size at the top and the bottom of the picture (ratio {ratio:.2f}): "
                     f"the camera may look straight down; posture read from box shape assumes a side view"}]


def warnings_text(warnings: list[dict]) -> str:
    """One line for the view warnings, plain ASCII ('' when there are none)."""
    return ("CAMERA VIEW WARNING (results less reliable): " + "; ".join(w["text"] for w in warnings)) if warnings else ""


def summary_text(problems: list[dict]) -> str:
    """One line: 'feed OK' or the problems, plain ASCII."""
    if not problems:
        return "feed OK: complete, not frozen, not dark"
    return "CAMERA FEED PROBLEM: " + "; ".join(p["text"] for p in problems)


def check(video_path, cfg: dict | None, tracker_result: dict | None = None) -> dict:
    """measure + assess (+ view_check when the detections are given). Never raises: when the check itself fails,
    the result says it was not checked."""
    if not _settings(cfg)["enabled"]:
        return {"ok": None, "problems": [], "warnings": [], "summary": "feed not checked (feed_health.enabled is off)"}
    try:
        res = assess(measure(video_path, cfg), cfg)
    except Exception as exc:                                       # never stop the run for an optional check
        res = {"ok": None, "problems": [], "warnings": [], "summary": f"feed not checked ({type(exc).__name__}: {exc})"}
    if tracker_result is not None and _settings(cfg)["view_warnings"]:
        try:
            res["warnings"] = res.get("warnings", []) + view_check(tracker_result, cfg)
        except Exception:                                          # a strange detections file: no view verdict
            pass
    return res


def site_check(cameras: list[dict], cfg: dict | None) -> dict:
    """Multi-camera: a camera whose recording ends more than end_missing_s before the latest one. cameras: dicts with
    "camera_id" (or "id"), "start_time" (ISO clock time) and "duration_s". Returns {camera_id: [problem]}."""
    import datetime as dt
    st = _settings(cfg)
    ends = {}
    for c in cameras:
        try:
            start = dt.datetime.fromisoformat(str(c.get("start_time")))
            ends[c.get("camera_id") or c.get("id")] = start + dt.timedelta(seconds=float(c.get("duration_s") or 0))
        except (TypeError, ValueError):
            continue
    if len(ends) < 2:
        return {}
    latest = max(ends.values())
    out = {}
    for cid, end in ends.items():
        short = (latest - end).total_seconds()
        if short >= float(st["end_missing_s"]):
            out[cid] = [{"kind": "ended_early", "start_s": None, "end_s": None,
                         "text": f"recording ends {short:.0f} s before the other cameras (feed lost?)"}]
    return out
