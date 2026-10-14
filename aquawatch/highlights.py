"""Highlight reel: a short, captioned video that contains only the incidents.

Instead of watching the whole clip, a guard watches highlights.mp4:
  1. a title card (2 s);
  2. every incident chain and every lone event, with a little context before and after
     (`highlights.pad_s`), overlapping ones merged and played in time order;
  3. a segment longer than `highlights.max_segment_s` is played fast-forward (every k-th
     processed frame) with a "FAST-FORWARD xk" tag, so the reel stays short;
  4. each frame is drawn by render.draw_frame, so it looks exactly like annotated.mp4
     (privacy blur first, then boxes, trails, labels), plus a caption bar:
     "Incident k/N | who BEHAVIOUR(S) | mm:ss-mm:ss | severity".

With nothing to show, the reel is just a card that says "No incidents".

Entry point: make_highlight_reel(video_path, tracker_result, tracks, final, zones, cfg, out_dir)
returns "highlights.mp4" (relative to out_dir) or None, and sets final["highlight_reel"].
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np

from render import (FONT, _ev_label, _event_color, _fit_scale, _label, _priority, _warn, _who, draw_frame,
                    make_context, transcode_h264)
from utils import ensure_dir, fmt_time

REEL_NAME = "highlights.mp4"
SEEK_MIN_FRAMES = 120            # jumping forward further than this seeks instead of decoding every frame
CARD_BG = (38, 30, 26)           # dark title card (BGR)
CARD_ACCENT = (60, 20, 220)
SEVERITY_BGR = {"high": (60, 60, 220), "medium": (0, 140, 255), "low": (40, 190, 230)}
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


# =========================================================================== planning

def _item_from_events(evs: list, start_s: float, end_s: float, severity, title, cfg, tracks) -> dict:
    """One 'incident' of the reel: a chain, or a single event that is not part of a chain."""
    evs = sorted(evs, key=lambda e: (e["start_s"], e.get("event_id", 0)))
    names: list[str] = []
    for e in evs:
        name = _ev_label(cfg, e)
        if name not in names:
            names.append(name)
    if severity is None:   # take the worst severity of the events
        ranked = [e.get("severity") for e in evs if e.get("severity") in SEVERITY_RANK]
        severity = max(ranked, key=SEVERITY_RANK.get) if ranked else None
    first = min(evs, key=_priority)
    return {"start_s": float(start_s), "end_s": float(end_s), "who": _who(evs[0], cfg, tracks),
            "behaviors": names, "severity": severity, "title": title,
            "event_ids": [e.get("event_id") for e in evs], "color": _event_color(first)}


def plan_segments(final: dict, tracker_result: dict, cfg: dict, tracks: dict | None = None) -> tuple[list, list]:
    """Decide what goes into the reel. Returns (incidents, segments); no video is touched.

    incidents: one dict per chain (final["incidents"]) and per event outside every chain, in time order:
      {start_s, end_s, who, behaviors, severity, title, event_ids, color}
    segments: [{start_s, end_s, items: [indices into incidents], ff}] after padding, clamping to the
      video length and merging overlaps. `ff` is the fast-forward factor (1 = normal speed).
    """
    tracks = tracks or {}
    hcfg = cfg.get("highlights") or {}
    pad = float(hcfg.get("pad_s", 1.5))
    max_seg = float(hcfg.get("max_segment_s", 8.0))
    events = (final or {}).get("events") or []
    by_id = {e.get("event_id"): e for e in events}

    items, in_chain = [], set()
    for ch in (final or {}).get("incidents") or []:
        evs = [by_id[i] for i in ch.get("event_ids", []) if i in by_id]
        if not evs:
            continue
        in_chain.update(e["event_id"] for e in evs)
        start = ch.get("start_s", min(e["start_s"] for e in evs))
        end = ch.get("end_s", max(e["end_s"] for e in evs))
        items.append(_item_from_events(evs, start, end, ch.get("severity"), ch.get("title"), cfg, tracks))
    for e in events:
        if e.get("event_id") not in in_chain:
            items.append(_item_from_events([e], e["start_s"], e["end_s"], e.get("severity"), None, cfg, tracks))
    items.sort(key=lambda it: (it["start_s"], it["end_s"]))

    duration = float((tracker_result or {}).get("duration_s") or 0.0)
    spans = []
    for idx, it in enumerate(items):
        hi = it["end_s"] + pad
        if duration > 0:
            hi = min(hi, duration)
        lo = min(max(0.0, it["start_s"] - pad), hi)
        spans.append((lo, hi, idx))
    segments: list[dict] = []
    for lo, hi, idx in sorted(spans):
        if segments and lo <= segments[-1]["end_s"]:
            segments[-1]["end_s"] = max(segments[-1]["end_s"], hi)
            segments[-1]["items"].append(idx)
        else:
            segments.append({"start_s": lo, "end_s": hi, "items": [idx]})
    for seg in segments:
        length = seg["end_s"] - seg["start_s"]
        seg["ff"] = max(1, int(math.ceil(length / max_seg - 1e-9))) if max_seg > 0 else 1
    return items, segments


# =========================================================================== drawing

def _centered(img, text: str, y: int, scale: float, color, thickness: int = 1) -> None:
    """Draw text horizontally centred, shrinking it if it would not fit."""
    w = img.shape[1]
    scale = _fit_scale(text, scale, thickness, 0.92 * w)
    (tw, _th), _base = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(img, text, ((w - tw) // 2, y), FONT, scale, color, thickness, cv2.LINE_AA)


def title_card(w: int, h: int, lines: list[tuple[str, float, tuple]]) -> np.ndarray:
    """A dark card with centred text lines [(text, scale at 640 px wide, colour BGR), ...]."""
    img = np.full((h, w, 3), CARD_BG, dtype=np.uint8)
    s = w / 640.0
    cv2.rectangle(img, (0, 0), (max(4, int(0.014 * w)), h), CARD_ACCENT, -1)
    heights = [int(sc * s * 34) for _t, sc, _c in lines]
    y = int(0.5 * (h - sum(heights)) + heights[0] * 0.8) if lines else 0
    for (text, sc, color), hh in zip(lines, heights):
        _centered(img, text, y, sc * s, color, max(1, int(round(sc * s * 1.4))))
        y += hh + int(10 * s)
    return img


def _caption(img, text: str, sub: str | None, severity: str | None) -> None:
    """Bottom caption bar: one text line, an optional smaller line, and a severity chip on the right."""
    h, w = img.shape[:2]
    s = max(w / 640.0, 0.6)
    thick = 1 if s < 1.5 else 2
    bar_h = int((56 if sub else 36) * s)
    y0 = max(0, h - bar_h)
    img[y0:] = (img[y0:] * 0.25 + np.array((30, 30, 30)) * 0.75).astype(np.uint8)

    chip_w = 0
    if severity:
        chip = severity.upper()
        cscale = 0.55 * s
        (cw, ch_), cbase = cv2.getTextSize(chip, FONT, cscale, thick)
        chip_w = cw + int(18 * s)
        x1, y_top = w - chip_w - int(8 * s), y0 + int(7 * s)
        col = SEVERITY_BGR.get(severity, (120, 120, 120))
        cv2.rectangle(img, (x1, y_top), (x1 + chip_w, y_top + ch_ + cbase + int(8 * s)), col, -1)
        cv2.putText(img, chip, (x1 + int(9 * s), y_top + ch_ + int(4 * s)), FONT, cscale,
                    (0, 0, 0) if severity == "low" else (255, 255, 255), thick, cv2.LINE_AA)
        chip_w += int(16 * s)
    scale = _fit_scale(text, 0.6 * s, thick, w - chip_w - 24 * s)
    (_tw, th), _b = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (int(12 * s), y0 + int(10 * s) + th), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)
    if sub:
        sscale = _fit_scale(sub, 0.5 * s, 1, w - 24 * s)
        cv2.putText(img, sub, (int(12 * s), h - int(10 * s)), FONT, sscale, (200, 200, 200), 1, cv2.LINE_AA)


def _current_item(items: list, idxs: list, t: float, tol: float) -> int:
    """Which incident of the segment the caption describes at time t: the active one, else the next one."""
    for i in idxs:
        if items[i]["start_s"] - tol <= t <= items[i]["end_s"] + tol:
            return i
    ahead = [i for i in idxs if items[i]["start_s"] > t]
    return ahead[0] if ahead else idxs[-1]


# =========================================================================== reading the video

class _Reader:
    """Frame reader that skips forward cheaply and seeks (verified) only for big jumps."""

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        self.pos = 0                         # index of the frame the next read()/grab() returns

    def opened(self) -> bool:
        return self.cap.isOpened()

    def _skip_to(self, frame: int) -> bool:
        while self.pos < frame:
            if not self.cap.grab():
                return False
            self.pos += 1
        return True

    def goto(self, frame: int) -> bool:
        """Make the next read() return `frame` (a frame at or after the current position is cheap)."""
        if frame >= self.pos and frame - self.pos <= SEEK_MIN_FRAMES:
            return self._skip_to(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
        if int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) == frame:
            self.pos = frame
            return True
        self.cap.release()                   # the seek was not exact: start over and decode up to the frame
        self.cap = cv2.VideoCapture(str(self.path))
        self.pos = 0
        return self._skip_to(frame)

    def read(self):
        ok, frame = self.cap.read()
        if ok:
            self.pos += 1
        return ok, frame

    def release(self) -> None:
        self.cap.release()


def _proc_size(tracker_result: dict, cfg: dict, reader: _Reader | None) -> tuple[int, int]:
    """Processing frame size (w, h), from the tracker result or, failing that, from the video itself."""
    proc = tracker_result.get("proc_size") or [0, 0]
    pw, ph = int(proc[0] or 0), int(proc[1] or 0)
    if pw > 0 and ph > 0:
        return pw, ph
    ow, oh = (tracker_result.get("orig_size") or [0, 0])[:2]
    if (not ow or not oh) and reader is not None and reader.opened():
        ow, oh = reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH), reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = pw or int(cfg["video"].get("resize_width") or 0) or int(ow or 640)
    return width, int(round(width * (oh or 360) / (ow or 640)))



# =========================================================================== the reel

def _write_segments(reader, writer, segments, items, ctx, size, last_frame) -> int:
    """Draw and write every segment. Returns the number of frames written."""
    w, h = size
    stride, fps = ctx.stride, ctx.fps
    n_items = len(items)
    written = 0
    for seg in segments:
        f0 = int(math.ceil(seg["start_s"] * fps / stride - 1e-6)) * stride
        f_last = int(math.floor(seg["end_s"] * fps / stride + 1e-6)) * stride
        if last_frame is not None:
            f_last = min(f_last, last_frame)
        ff = seg["ff"]
        for m in range((f_last - f0) // stride + 1):
            if m % ff != 0:
                continue                     # fast-forward: never decode the frames we do not show
            f = f0 + m * stride
            if not reader.goto(f):
                break
            ok, frame = reader.read()
            if not ok:
                break
            t = f / fps
            img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            draw_frame(img, f, t, ctx)
            k = _current_item(items, seg["items"], t, ctx.tol)
            it = items[k]
            text = (f"Incident {k + 1}/{n_items} | {it['who']} {' > '.join(it['behaviors'])}"
                    f" | {fmt_time(it['start_s'])}-{fmt_time(it['end_s'])}")
            _caption(img, text, it.get("title"), it.get("severity"))
            if ff > 1:
                s = max(w / 640.0, 0.6)
                _label(img, f"FAST-FORWARD x{ff}", w, int(32 * s), (0, 165, 255), scale=0.6 * s,
                       thickness=1 if s < 1.5 else 2)
            writer.write(img)
            written += 1
    return written


def _make_reel(video_path, tracker_result, tracks, final, zones, cfg, out: Path) -> str | None:
    hcfg = cfg.get("highlights") or {}
    ctx = make_context(tracker_result, tracks, final, zones, cfg)
    items, segments = plan_segments(final, tracker_result, cfg, tracks)
    out_fps = ctx.fps / ctx.stride

    reader = _Reader(video_path) if segments else None
    if segments and not reader.opened():
        _warn(f"cannot open video {video_path}; no highlight reel")
        return None
    pw, ph = _proc_size(tracker_result, cfg, reader)
    w, h = max(2, pw - pw % 2), max(2, ph - ph % 2)         # H.264 / yuv420p need even sizes

    tmp = out / "highlights_tmp.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
    if not writer.isOpened():
        _warn("cv2.VideoWriter could not open an mp4v file; no highlight reel")
        if reader is not None:
            reader.release()
        return None

    try:
        # title card
        seconds = float(tracker_result.get("duration_s") or 0.0) or max([it["end_s"] for it in items] + [0.0])
        video_name = Path(str(video_path)).name
        scenario = ((cfg.get("scenario") or {}).get("title") or "").strip()
        white, grey = (255, 255, 255), (190, 190, 190)
        if items:
            n = len(items)
            lines = [("HIGHLIGHT REEL", 1.5, white), (video_name, 0.8, grey),
                     (f"{n} incident{'s' if n != 1 else ''} in {fmt_time(seconds)} of video", 0.9, white)]
        else:
            lines = [("HIGHLIGHT REEL", 1.1, grey), ("No incidents", 1.6, white),
                     (f"{video_name} ({fmt_time(seconds)})", 0.7, grey)]
        if scenario:
            lines.append((scenario, 0.6, grey))
        if ctx.privacy:
            lines.append(("Privacy mode: faces blurred", 0.6, (0, 190, 255)))
        card = title_card(w, h, lines)
        for _ in range(max(1, int(round(float(hcfg.get("title_s", 2.0)) * out_fps)))):
            writer.write(card)

        if segments:
            limit = int(tracker_result.get("n_frames_read") or 0)
            last_frame = ((limit - 1) // ctx.stride) * ctx.stride if limit > 0 else None
            _write_segments(reader, writer, segments, items, ctx, (w, h), last_frame)
    finally:
        writer.release()
        if reader is not None:
            reader.release()

    dst = out / REEL_NAME
    if tmp.stat().st_size == 0:
        tmp.unlink()
        return None
    if transcode_h264(tmp, dst):
        tmp.unlink()
    else:                                                    # no ffmpeg (or it failed): keep the mp4v file
        os.replace(tmp, dst)
    return REEL_NAME


def make_highlight_reel(video_path, tracker_result: dict, tracks: dict, final: dict, zones: list,
                        cfg: dict, out_dir) -> str | None:
    """Write highlights.mp4 into out_dir and return its name, or None if it was not made.

    Sets final["highlight_reel"] to the same value. Never raises: a failure here only costs the
    reel, never the events or the report.
    """
    if final is not None:
        final["highlight_reel"] = None
    if not (cfg.get("highlights") or {}).get("enabled", True):
        return None
    out = ensure_dir(out_dir)
    try:
        name = _make_reel(video_path, tracker_result or {}, tracks or {}, final or {}, zones or [], cfg, out)
    except Exception as exc:  # noqa: BLE001 - optional step
        _warn(f"highlight reel failed: {type(exc).__name__}: {exc}")
        name = None
        try:
            (out / "highlights_tmp.mp4").unlink()
        except OSError:
            pass
    if final is not None:
        final["highlight_reel"] = name
    return name
