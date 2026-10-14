"""Per-object tracks: group detections by ID, smooth the foot point, measure speed and dwell time.

NumPy only (no OpenCV / PyTorch), so it can be unit-tested anywhere.

Everything is measured in body-scale units so one set of thresholds works for objects near and far
from the camera:
  * foot point  = bottom-centre of the box, ((x1 + x2) / 2, y2)
  * body scale  = max(box height, box width). For an upright person this is the box height (the old
                  "body-height"); for a lying person or a car it is the body length, so speeds and
                  distances stay sane when the box turns sideways. Set `features.scale: box_h` to
                  go back to plain box height.
  * speed       = foot displacement per second, divided by the body scale (bh/s)
  * dwell       = how long the object stayed inside a small circle (radius in body-scale units)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from utils import load_config, moving_average

_EPS = 1e-9            # float tolerance for comparing times that came from frame_idx / fps
EDGE_MARGIN_PX = 3     # a box within this many pixels of the frame border counts as "cut off"
ASPECT_CAP = 10.0      # w/h is capped here so a sliver-thin box cannot dominate the smoothing


# --------------------------------------------------------------------------- data model

@dataclass(eq=False)  # eq=False: comparing NumPy-array fields with == is ambiguous, so use identity
class Track:
    """One tracked object (one tracker ID) over time. All arrays have the same length n.

    The last four fields are optional so a Track can still be built by hand (as the tests do):
    `cls` defaults to 0 (person), the arrays default to "unknown" (NaN / False).
    """
    track_id: int
    t: np.ndarray        # (n,) seconds, strictly increasing
    frame: np.ndarray    # (n,) int original frame indices
    box: np.ndarray      # (n, 4) raw x1, y1, x2, y2 (processing px)
    conf: np.ndarray     # (n,) detection confidence
    foot: np.ndarray     # (n, 2) SMOOTHED foot point (moving average, per segment)
    height: np.ndarray   # (n,) SMOOTHED body scale in px (max(box h, box w); = body-height for upright people)
    speed: np.ndarray    # (n,) bh/s, NaN where unknown (start of a segment or across a gap)
    segment: np.ndarray  # (n,) int, +1 at every time gap > features.max_gap_s
    cls: int = 0                       # most common class id of the track (COCO numbering, 0 = person)
    aspect: np.ndarray | None = None   # (n,) SMOOTHED box width / height (lying person: > 1)
    vel: np.ndarray | None = None      # (n, 2) SMOOTHED foot velocity in px/s, NaN where unknown
    at_edge: np.ndarray | None = None  # (n,) bool, True when the box touches (or is near) the frame border
    frame_size: tuple | None = None    # (w, h) of the processing frame (used by the approaching alert)

    def __post_init__(self) -> None:
        """Fill the optional arrays with "unknown" values when the caller did not give them."""
        n = len(self.t)
        if self.aspect is None:
            self.aspect = _raw_aspect(np.asarray(self.box, dtype=float).reshape(-1, 4)) if n else np.zeros(0)
        if self.vel is None:
            self.vel = np.full((n, 2), np.nan)
        if self.at_edge is None:
            self.at_edge = np.zeros(n, dtype=bool)
        self.cls = int(self.cls)

    @property
    def start_s(self) -> float:
        """Time of the first sample."""
        return float(self.t[0]) if len(self.t) else 0.0

    @property
    def end_s(self) -> float:
        """Time of the last sample."""
        return float(self.t[-1]) if len(self.t) else 0.0

    @property
    def duration_s(self) -> float:
        """Seconds between the first and the last sample."""
        return self.end_s - self.start_s


# --------------------------------------------------------------------------- building tracks

def build_tracks(tracker_result: dict, cfg: dict) -> dict[int, Track]:
    """Turn the raw detection list into one smoothed Track per tracker ID.

    Steps per ID: sort by time (duplicates keep the highest confidence), drop tiny boxes,
    drop IDs that are too short, split at time gaps, smooth the foot point, body scale and
    box aspect inside each segment, then compute speed (bh/s) and velocity (px/s).
    """
    fcfg = cfg["features"]
    dets = _detections_array(tracker_result)
    tracks: dict[int, Track] = {}
    if len(dets) == 0:
        return tracks
    proc_size = (tracker_result or {}).get("proc_size")
    # min_box_h_px is meant in ORIGINAL video pixels: the same person must not count as 'too small' on one camera and
    # not on another just because the videos are resized to different scales (bug hunt 2026-10-11, tracking high;
    # PREREGISTRATION_bug_group_C.md step C1)
    orig_size = (tracker_result or {}).get("orig_size")
    to_proc = float(proc_size[0]) / float(orig_size[0]) if proc_size and orig_size and float(orig_size[0]) else 1.0
    ids = np.rint(dets[:, 2]).astype(int)
    for tid in np.unique(ids):                 # np.unique is sorted, so the dict is ordered by ID
        track = _make_track(int(tid), dets[ids == tid], fcfg, proc_size, to_proc)
        if track is not None:
            track.frame_size = (float(proc_size[0]), float(proc_size[1])) if proc_size else None
            tracks[int(tid)] = track
    return tracks


def _detections_array(tracker_result: dict) -> np.ndarray:
    """Detections as an (n, 9) float array [frame, t, id, x1, y1, x2, y2, conf, cls]; bad rows removed.

    8-element rows (old files) get class 0; a row with fewer than 8 numbers is dropped.
    """
    rows = (tracker_result or {}).get("detections") or []
    if len(rows) == 0:
        return np.empty((0, 9), dtype=float)
    try:
        arr = np.asarray(rows, dtype=float)
    except ValueError:                                      # rows of different lengths (8 and 9 mixed)
        arr = np.asarray([list(r[:9]) + [0.0] * (9 - len(r)) for r in rows if len(r) >= 8], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 8 or len(arr) == 0:
        return np.empty((0, 9), dtype=float)
    if arr.shape[1] == 8:
        arr = np.column_stack((arr, np.zeros(len(arr))))     # class 0 = person
    arr = arr[:, :9]
    arr = arr[np.all(np.isfinite(arr[:, :8]), axis=1)]
    arr[~np.isfinite(arr[:, 8]), 8] = 0.0
    return arr


def _raw_aspect(box: np.ndarray) -> np.ndarray:
    """Box width / height for every row of an (n, 4) x1, y1, x2, y2 array (NaN for an empty box)."""
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]
    out = np.full(len(box), np.nan)
    ok = (h > 0) & (w >= 0)
    out[ok] = np.minimum(w[ok] / h[ok], ASPECT_CAP)
    return out


def _body_scale(box: np.ndarray, mode: str) -> np.ndarray:
    """Body size per box in px: max(height, width) by default, or just the height (`scale: box_h`)."""
    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    return h if str(mode).lower() in ("box_h", "height", "box_height") else np.maximum(h, w)


def _make_track(track_id: int, rows: np.ndarray, fcfg: dict, proc_size=None, to_proc: float = 1.0) -> Track | None:
    """Build one Track from that ID's detection rows, or None if the ID is not usable."""
    # Sort by time; for duplicate times put the highest confidence first and keep only that one.
    order = np.lexsort((-rows[:, 7], rows[:, 1]))
    rows = rows[order]
    first_of_time = np.ones(len(rows), dtype=bool)
    first_of_time[1:] = np.diff(rows[:, 1]) > 0
    rows = rows[first_of_time]

    # Tiny boxes (object too far away) cannot be measured reliably.
    scale_mode = fcfg.get("scale", "max_side")
    rows = rows[_body_scale(rows[:, 3:7], scale_mode) >= float(fcfg["min_box_h_px"]) * to_proc]
    if len(rows) < 3:
        return None
    t = rows[:, 1].copy()
    if t[-1] - t[0] < float(fcfg["min_track_s"]):
        return None

    box = rows[:, 3:7].copy()
    foot = np.column_stack(((box[:, 0] + box[:, 2]) / 2.0, box[:, 3]))   # bottom-centre of the box
    scale = _body_scale(box, scale_mode)
    classes, counts = np.unique(np.rint(rows[:, 8]).astype(int), return_counts=True)
    cls = int(classes[int(np.argmax(counts))])                            # most common class id

    # A time gap longer than max_gap_s starts a new segment (the object was lost for a while).
    segment = np.zeros(len(t), dtype=int)
    segment[1:] = np.cumsum(np.diff(t) > float(fcfg["max_gap_s"]))

    # Smooth inside each segment so the average never mixes samples from before and after a gap.
    # All five signals (foot x, foot y, body scale, aspect, "touches the frame border") share one pass.
    signals = np.column_stack((foot, scale, _raw_aspect(box), _touches_edge(box, proc_size).astype(float)))
    smooth = _smooth_by_segment(signals, t, segment, float(fcfg["smoothing_s"]))
    foot_s, scale_s, aspect_s = smooth[:, 0:2].copy(), smooth[:, 2].copy(), smooth[:, 3].copy()
    edge = smooth[:, 4] > 0                       # any border contact within the smoothing window
    speed = _compute_speed(t, foot_s, scale_s, segment, float(fcfg["speed_window_s"]))
    vel = _compute_velocity(t, foot_s, segment, float(fcfg["speed_window_s"]))

    return Track(track_id=track_id, t=t, frame=np.rint(rows[:, 0]).astype(int), box=box,
                 conf=rows[:, 7].copy(), foot=foot_s, height=scale_s, speed=speed, segment=segment,
                 cls=cls, aspect=aspect_s, vel=vel, at_edge=edge)


def _touches_edge(box: np.ndarray, proc_size) -> np.ndarray:
    """True for boxes that touch the frame border (within EDGE_MARGIN_PX): they are cut off, so their
    shape says little about the body. Without a known frame size nothing counts as cut off."""
    if not proc_size or len(proc_size) < 2:
        return np.zeros(len(box), dtype=bool)
    width, height = float(proc_size[0]), float(proc_size[1])
    m = EDGE_MARGIN_PX
    return (box[:, 0] <= m) | (box[:, 1] <= m) | (box[:, 2] >= width - m) | (box[:, 3] >= height - m)


def _smooth_by_segment(values: np.ndarray, t: np.ndarray, segment: np.ndarray, window_s: float) -> np.ndarray:
    """utils.moving_average applied to each segment on its own."""
    out = np.empty_like(values, dtype=float)
    for seg in np.unique(segment):
        sel = segment == seg
        out[sel] = moving_average(values[sel], t[sel], window_s)
    return out


def window_start(t: np.ndarray, segment: np.ndarray, window_s: float) -> np.ndarray:
    """For every sample i, the index of the earliest sample j (same segment) with t[j] >= t[i] - window_s.

    Shared by the speed, the velocity and the closing-speed calculations so they all look back over
    the same time window and never reach across a tracking gap.
    """
    seg_first = np.searchsorted(segment, segment, side="left")           # first index of own segment
    lo = np.searchsorted(t, t - window_s - _EPS, side="left")             # earliest t[j] >= t[i] - window
    return np.maximum(lo, seg_first)


def _compute_speed(t: np.ndarray, foot: np.ndarray, height: np.ndarray, segment: np.ndarray,
                   window_s: float) -> np.ndarray:
    """Speed in bh/s at every sample: displacement over the last `window_s` seconds / elapsed / scale.

    NaN when fewer than half a window of history exists inside the same segment (the start of a
    track or of the stretch after a gap), because a speed from a tiny time span would be noise.
    """
    speed = np.full(len(t), np.nan)
    lo = window_start(t, segment, window_s)
    dt = t - t[lo]
    dist = np.hypot(foot[:, 0] - foot[lo, 0], foot[:, 1] - foot[lo, 1])
    ok = (dt >= 0.5 * window_s - _EPS) & (dt > 0) & (height > 0)
    speed[ok] = dist[ok] / dt[ok] / height[ok]
    return speed


def _compute_velocity(t: np.ndarray, foot: np.ndarray, segment: np.ndarray, window_s: float) -> np.ndarray:
    """Foot velocity (vx, vy) in px/s over the last `window_s` seconds; NaN where speed would be NaN."""
    vel = np.full((len(t), 2), np.nan)
    lo = window_start(t, segment, window_s)
    dt = t - t[lo]
    ok = (dt >= 0.5 * window_s - _EPS) & (dt > 0)
    vel[ok] = (foot[ok] - foot[lo[ok]]) / dt[ok, None]
    return vel


# --------------------------------------------------------------------------- dwell / stillness

def window_radius_bh(foot: np.ndarray, height: np.ndarray) -> float:
    """Spread of a group of foot points: the largest distance from their centroid, in body-scale units.

    The scale is the median body size of the group. Used by the loitering rule and the
    loitering metrics so both measure "how much did the object move" the same way.
    """
    centroid = foot.mean(axis=0)
    dist = np.hypot(foot[:, 0] - centroid[0], foot[:, 1] - centroid[1])
    bh = float(np.median(height))
    return float(dist.max() / bh) if bh > 0 else float("inf")


def loiter_segments(track: Track, radius_bh: float, bridge_gap_s: float = 0.0) -> np.ndarray:
    """Segment ids for STANDING-STILL rules: like track.segment, but a short dropout is bridged.

    If the object disappears for at most `bridge_gap_s` (missed detections, hidden behind a pillar)
    and reappears within `radius_bh` of where it was, both pieces count as one stationary stretch.
    Speeds are still split at every gap (they would be wrong across it).
    """
    seg = np.asarray(track.segment).copy()
    if bridge_gap_s <= 0 or len(seg) == 0:
        return seg
    merged = seg.copy()
    for k in range(1, len(seg)):
        if seg[k] == seg[k - 1]:
            merged[k] = merged[k - 1]
            continue
        gap = float(track.t[k] - track.t[k - 1])
        scale = 0.5 * float(track.height[k] + track.height[k - 1])
        moved = float(np.hypot(*(track.foot[k] - track.foot[k - 1])))
        same_spot = scale > 0 and moved <= radius_bh * scale
        merged[k] = merged[k - 1] if (gap <= bridge_gap_s and same_spot) else merged[k - 1] + 1
    return merged


def stationary_runs(track: Track, radius_bh: float, bridge_gap_s: float = 0.0) -> float:
    """Longest time (s) the object stayed inside a circle of `radius_bh` body-scale units.

    A span counts when every smoothed foot point in it is within radius_bh * median(scale in span)
    of the span's centroid. Works inside segments only (a gap breaks a span). Implemented with a
    sliding window: for each end sample the start is pushed forward until the span fits the circle,
    so the cost is a few NumPy operations per sample.
    """
    best = 0.0
    segments = loiter_segments(track, radius_bh, bridge_gap_s)
    for seg in np.unique(segments):
        idx = np.flatnonzero(segments == seg)
        if len(idx) < 2:
            continue
        t, foot, height = track.t[idx], track.foot[idx], track.height[idx]
        j = 0
        for i in range(len(idx)):
            while j < i and window_radius_bh(foot[j:i + 1], height[j:i + 1]) > radius_bh:
                j += 1
            best = max(best, float(t[i] - t[j]))
    return best


# --------------------------------------------------------------------------- summary

@lru_cache(maxsize=1)
def _default_cfg() -> dict:
    """The project's config.yaml (cached); only used when the caller does not pass a config."""
    return load_config()


def _bridge_gap(cfg: dict | None) -> float:
    """loitering.bridge_gap_s from the config (0 = never bridge)."""
    cfg = cfg or _default_cfg()
    return float(((cfg.get("behaviors") or {}).get("loitering") or {}).get("bridge_gap_s", 0.0))


def track_summary(track: Track, cfg: dict | None = None) -> dict:
    """One-line description of a track for events.json and the report.

    `cfg` is optional: pass it so max_dwell_s uses the same loitering radius as the run; without
    it the project's config.yaml is used. Speeds are 0.0 when the track never had a valid speed.
    """
    radius = float((cfg or _default_cfg())["behaviors"]["loitering"]["radius_bh"])
    speeds = track.speed[np.isfinite(track.speed)]
    median = float(np.median(speeds)) if len(speeds) else 0.0
    p90 = float(np.percentile(speeds, 90)) if len(speeds) else 0.0
    return {
        "id": int(track.track_id),
        "first_seen": round(track.start_s, 2),
        "last_seen": round(track.end_s, 2),
        "duration_s": round(track.duration_s, 2),
        "median_speed_bh_s": round(median, 3),
        "p90_speed_bh_s": round(p90, 3),
        "max_dwell_s": round(stationary_runs(track, radius, _bridge_gap(cfg)), 2),
    }
