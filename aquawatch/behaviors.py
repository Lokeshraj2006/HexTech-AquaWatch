"""Behaviour rules: loitering, zone intrusion, running, fall, crowding, near miss + the "normal" baseline.

NumPy only. The rules are deliberately simple and explainable: each one is a measurable condition
(distance, time, speed, shape) with thresholds from config.yaml, so every flagged event can be
justified with numbers.

`detect_behaviors` returns RAW intervals (not merged, not duration-filtered for running / zone / fall
/ crowding). events.py merges close pieces, applies the minimum durations and builds the final events.

Three kinds of rule:
  * one object over time ..... loitering, zone intrusion, running, fall
  * a scene-level count ....... crowding (how many objects are inside a zone at the same time)
  * a pair of objects ......... near miss (two objects closed in on each other while moving)
"""
from __future__ import annotations

import math

import numpy as np

import approaching

from features import Track, stationary_runs, window_radius_bh, window_start, loiter_segments
from utils import (BEHAVIORS, CLASS_NAMES, distance_to_polygon_edge, entity_name, entity_word,
                   mask_to_intervals, point_in_polygon, robust_z, state_classes, unit_name)

_EPS = 1e-6
BASELINE_MIN_TRACK_S = 2.0   # a track must last this long to count as "a typical person" in the baseline
LOITER_COVERAGE = 0.9        # a loitering window must really be observed for >= 90% of min_duration_s
FALL_PEAK_OFFSET_S = 0.5     # a fall's snapshot is taken this long after the person goes down
NEAR_PAD_S = 0.25            # a near miss seen in a single sample is widened by this much on each side
APPROACH_TOL_BH = 0.02       # noise allowed (in body-scale units) when measuring how long a distance was falling
WHOLE_FRAME = "whole frame"  # zone name used by crowding when no zone is drawn and whole_frame is on
VEHICLE_CLASSES = (1, 2, 3, 5, 7)   # bicycle, car, motorbike, bus, truck: wider than tall by nature,
                                    # so the lying-down rule does not judge them (override: fall.classes)


# --------------------------------------------------------------------------- small helpers

def _rule(cfg: dict, key: str) -> dict:
    """The config section of a rule; an empty, disabled rule when the config does not have it."""
    return (cfg.get("behaviors") or {}).get(key) or {"enabled": False}


def _zone_name(zone: dict, k: int) -> str:
    """Zone name, with a fallback so unnamed zones still have a label."""
    return str(zone.get("name") or f"zone{k + 1}")


def _short_unit(cfg: dict) -> str:
    """Abbreviation of the size unit: "body-heights" -> "bh", "body-lengths" -> "bl"."""
    return "".join(word[0] for word in unit_name(cfg).split("-") if word) or "u"


def _inside_mask(track: Track, points) -> np.ndarray:
    """True for every sample whose smoothed foot point is inside the polygon (None = the whole frame)."""
    if points is None:
        return np.ones(len(track.t), dtype=bool)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    # Cheap bounding-box test first, so the polygon test only runs for nearby samples.
    near = ((track.foot[:, 0] >= min(xs)) & (track.foot[:, 0] <= max(xs)) &
            (track.foot[:, 1] >= min(ys)) & (track.foot[:, 1] <= max(ys)))
    inside = np.zeros(len(track.t), dtype=bool)
    for i in np.flatnonzero(near):
        inside[i] = point_in_polygon(float(track.foot[i, 0]), float(track.foot[i, 1]), points)
    return inside


def _span_indices(track: Track, start_s: float, end_s: float) -> np.ndarray:
    """Indices of the samples inside [start_s, end_s]; the nearest sample if none fall inside."""
    return _times_in_span(track.t, start_s, end_s)


def _times_in_span(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Indices of the times inside [start_s, end_s]; the nearest one if none fall inside."""
    idx = np.flatnonzero((t >= start_s - _EPS) & (t <= end_s + _EPS))
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(t - 0.5 * (start_s + end_s))))])
    return idx


# --------------------------------------------------------------------------- rules for one object

def loitering_intervals(track: Track, cfg: dict) -> list[tuple[float, float]]:
    """Time spans where the object stayed within `radius_bh` body-scale units of one spot.

    For every sample i we look at the window [t[i] - min_duration_s, t[i]] inside one segment.
    If the window was really observed (covers >= 90% of min_duration_s) and every smoothed foot
    point in it is within radius_bh * median(scale) of the window centroid, all samples of the
    window are marked. Runs of marked samples are the loitering intervals.
    """
    lcfg = cfg["behaviors"]["loitering"]
    window_s = float(lcfg["min_duration_s"])
    radius = float(lcfg["radius_bh"])
    t = track.t
    mask = np.zeros(len(t), dtype=bool)
    bridge = float(lcfg.get("bridge_gap_s", 0.0))
    segments = loiter_segments(track, radius, bridge)          # short dropouts do not reset the clock
    seg_first = np.searchsorted(segments, segments, side="left")
    win_first = np.maximum(np.searchsorted(t, t - window_s - _EPS, side="left"), seg_first)
    for i in range(len(t)):
        j = int(win_first[i])
        if t[i] - t[j] < LOITER_COVERAGE * window_s:
            continue                                   # not enough history yet (or a gap)
        if window_radius_bh(track.foot[j:i + 1], track.height[j:i + 1]) <= radius:
            mask[j:i + 1] = True
    runs = mask_to_intervals(t, mask, max(bridge, float(cfg["features"]["max_gap_s"])))
    return [(s, e) for s, e, _, _ in runs]


def zone_intervals(track: Track, zones: list[dict], cfg: dict) -> list[tuple[str, float, float]]:
    """Time spans where the smoothed foot point was inside a zone polygon: [(zone_name, start, end)].

    Every sample is assigned to the first zone that contains it, then each run of samples in the same
    zone is one interval. No duration filter here (events.py filters after merging).
    """
    t = track.t
    zone_of = np.full(len(t), -1, dtype=int)
    for k, zone in enumerate(zones):
        inside = _inside_mask(track, zone["points"])
        zone_of[inside & (zone_of < 0)] = k
    out = []
    for k, zone in enumerate(zones):
        for s, e, _, _ in mask_to_intervals(t, zone_of == k, float(cfg["features"]["max_gap_s"])):
            out.append((_zone_name(zone, k), s, e))
    return out


def fall_applies(track: Track, cfg: dict) -> bool:
    """Does the lying-down rule judge this track? Needs fall.enabled and a class that stands upright.

    Vehicles are always wider than tall, so they are skipped. A preset can list the classes to judge
    explicitly with `fall.classes`.
    """
    fcfg = _rule(cfg, "fall")
    if not fcfg.get("enabled"):
        return False
    classes = fcfg.get("classes")
    if classes:
        return int(track.cls) in {int(c) for c in classes}
    return int(track.cls) not in VEHICLE_CLASSES


def down_mask(track: Track, cfg: dict) -> np.ndarray:
    """True for the samples where the object is lying: aspect (w/h) >= down_min_aspect.

    A box that touches the frame border does not count (it is cut off, so its shape is not the body's).
    All False when the fall rule is off or does not apply to this class.
    """
    if not fall_applies(track, cfg):
        return np.zeros(len(track.t), dtype=bool)
    threshold = float(cfg["behaviors"]["fall"]["down_min_aspect"])
    with np.errstate(invalid="ignore"):                     # NaN aspect (unknown box shape) compares False
        return np.isfinite(track.aspect) & (track.aspect >= threshold) & ~track.at_edge


def fall_intervals(track: Track, cfg: dict) -> list[tuple[float, float]]:
    """Runs of samples where the object is down. No duration filter here (events.py applies min_down_s)."""
    runs = mask_to_intervals(track.t, down_mask(track, cfg), float(cfg["features"]["max_gap_s"]))
    return [(s, e) for s, e, _, _ in runs]


def running_intervals(track: Track, cfg: dict) -> list[tuple[float, float]]:
    """Time spans of running, using hysteresis on the speed signal.

    The state switches ON when speed >= start_speed_bh_s and OFF when speed < end_speed_bh_s
    (end < start), so a person hovering around the threshold does not flicker on and off.
    NaN speeds change nothing; the state resets at every new segment (after a tracking gap).
    Samples where the object is down are never running (a fall's speed spike is not a run): they
    switch the state off.
    """
    rcfg = cfg["behaviors"]["running"]
    start_thr, end_thr = float(rcfg["start_speed_bh_s"]), float(rcfg["end_speed_bh_s"])
    down = down_mask(track, cfg)
    on = np.zeros(len(track.t), dtype=bool)
    state, prev_seg = False, None
    for i in range(len(track.t)):
        if track.segment[i] != prev_seg:
            state, prev_seg = False, track.segment[i]
        if down[i]:
            state = False
            continue
        v = track.speed[i]
        if math.isfinite(v):
            if not state and v >= start_thr:
                state = True
            elif state and v < end_thr:
                state = False
        on[i] = state
    runs = mask_to_intervals(track.t, on, float(cfg["features"]["max_gap_s"]))
    return [(s, e) for s, e, _, _ in runs]


def _max_speed(track: Track) -> float:
    """Highest valid speed of the track in bh/s (0.0 if none)."""
    v = track.speed[np.isfinite(track.speed)]
    return float(v.max()) if len(v) else 0.0


def _median_speed(track: Track) -> float:
    """Median valid speed of the track in bh/s (NaN if none)."""
    v = track.speed[np.isfinite(track.speed)]
    return float(np.median(v)) if len(v) else float("nan")


# --------------------------------------------------------------------------- crowding (scene-level count)

def _crowd_zones(zones: list[dict], cfg: dict) -> list[tuple[str, list | None]]:
    """The areas people are counted in: each drawn zone, or the whole frame (only if no zone is drawn
    and crowding.whole_frame is on). Returns [(zone_name, polygon points or None for the whole frame)]."""
    if zones:
        return [(_zone_name(z, k), z["points"]) for k, z in enumerate(zones)]
    if _rule(cfg, "crowding").get("whole_frame"):
        return [(WHOLE_FRAME, None)]
    return []


def crowd_profile(tracks: dict[int, Track], polygon, cfg: dict) -> dict:
    """How many tracks are inside `polygon` (None = anywhere) at every processed time.

    Returns {"t": times, "count": heads per time, "inside": {track_id: bool per time}, "conf_sum":
    summed detection confidence of the counted tracks per time}. A track that is briefly missing
    inside one segment (a few dropped detections) keeps its last known state, so one lost frame does
    not make the count flicker. Times across a real tracking gap count as absent.
    """
    if not tracks:
        return {"t": np.zeros(0), "count": np.zeros(0, dtype=int), "inside": {}, "conf_sum": np.zeros(0)}
    all_t = np.unique(np.concatenate([tr.t for tr in tracks.values()]))
    count = np.zeros(len(all_t), dtype=int)
    conf_sum = np.zeros(len(all_t))
    inside: dict[int, np.ndarray] = {}
    for tid in sorted(tracks):
        tr = tracks[tid]
        own = _inside_mask(tr, polygon)
        if not own.any():
            continue
        k = np.searchsorted(tr.t, all_t + _EPS, side="right") - 1          # latest own sample at or before
        seen = k >= 0
        k = np.maximum(k, 0)
        nxt = np.minimum(k + 1, len(tr.t) - 1)
        exact = np.abs(all_t - tr.t[k]) <= _EPS
        bridged = (k < len(tr.t) - 1) & (tr.segment[k] == tr.segment[nxt])  # between two samples of one segment
        here = seen & (exact | bridged) & own[k]
        inside[tid] = here
        count += here
        conf_sum += np.where(here, tr.conf[k], 0.0)
    return {"t": all_t, "count": count, "inside": inside, "conf_sum": conf_sum}


def crowding_intervals(tracks: dict[int, Track], zones: list[dict], cfg: dict) -> list[dict]:
    """Raw crowding intervals: runs of time where >= crowding.min_count tracks are inside one zone.

    Each interval has track_id None (it belongs to the scene, not to one object), the zone name and
    `entities`: the ids that were inside during the run.
    """
    ccfg = _rule(cfg, "crowding")
    min_count = int(ccfg.get("min_count", 5))
    out: list[dict] = []
    if len(tracks) < min_count:
        return out
    for name, polygon in _crowd_zones(zones, cfg):
        prof = crowd_profile(tracks, polygon, cfg)
        for s, e, i0, i1 in mask_to_intervals(prof["t"], prof["count"] >= min_count,
                                              float(cfg["features"]["max_gap_s"])):
            members = [tid for tid, here in prof["inside"].items() if here[i0:i1 + 1].any()]
            out.append(_interval(None, "crowding", s, e, zone=name, entities=members))
    return out


def crowding_metrics(tracks: dict[int, Track], zones: list[dict], start_s: float, end_s: float,
                     cfg: dict, zone_name: str | None = None) -> tuple[dict, list[int]]:
    """Numbers for one (merged) crowding span, plus the ids that were inside during it.

    Needs ALL tracks (the count is a property of the scene). Returns (metrics, entities).
    """
    areas = _crowd_zones(zones, cfg)
    polygon = next((p for n, p in areas if n == zone_name), areas[0][1] if areas else None)
    name = zone_name or (areas[0][0] if areas else WHOLE_FRAME)
    prof = crowd_profile(tracks, polygon, cfg)
    idx = _times_in_span(prof["t"], start_s, end_s) if len(prof["t"]) else np.zeros(0, dtype=int)
    count = prof["count"][idx] if len(idx) else np.zeros(1, dtype=int)
    entities = sorted(tid for tid, here in prof["inside"].items() if len(idx) and here[idx].any())
    counted = float(count.sum())
    mean_conf = float(prof["conf_sum"][idx].sum() / counted) if counted > 0 else 0.0
    peak = float(prof["t"][idx][int(np.argmax(count))]) if len(idx) else 0.5 * (start_s + end_s)
    metrics = {
        "duration_s": round(max(0.0, end_s - start_s), 3),
        "zone": name,
        "max_count": int(count.max()),
        "mean_count": round(float(count.mean()), 2),
        "min_count": int(count.min()),
        "min_duration_s": float(_rule(cfg, "crowding").get("min_duration_s", 0.0)),
        "peak_time_s": round(peak, 3),
        "mean_det_conf": round(mean_conf, 3),
    }
    return metrics, entities


# --------------------------------------------------------------------------- near miss (pairs of objects)

def near_miss_pairs(tracks: dict[int, Track], cfg: dict) -> list[tuple[Track, Track]]:
    """Which pairs (A, B) are checked: A is vulnerable (a person by default), B is anything in other_classes.

    A pair of two vulnerable objects is checked once, with A = the smaller id. Pairs that were never
    visible at the same time are skipped.
    """
    ncfg = _rule(cfg, "near_miss")
    vulnerable = {int(c) for c in ncfg.get("vulnerable_classes", [0])}
    others = {int(c) for c in ncfg.get("other_classes", [])}
    pairs = []
    for a_id in sorted(tracks):
        a = tracks[a_id]
        if a.cls not in vulnerable:
            continue
        for b_id in sorted(tracks):
            b = tracks[b_id]
            if b_id == a_id or b.cls not in others:
                continue
            if b.cls in vulnerable and b_id < a_id:
                continue                                     # this people pair is checked as (b, a)
            if b.end_s < a.start_s or a.end_s < b.start_s:
                continue                                     # never on screen together
            pairs.append((a, b))
    return pairs


def pair_signals(a: Track, b: Track, cfg: dict) -> dict | None:
    """The time series that describe how A and B moved relative to each other (DESIGN 4.8).

    Only times where both were detected on the same frame are used. All distances and speeds are in
    body-scale units: A's scale, or the mean of both scales when both are vulnerable (two people).
      d     distance between the smoothed foot points
      rel   speed of A relative to B (|velA - velB|)
      vc    closing speed: how fast d is falling over speed_window_s (positive = approaching)
      ttc   time to collision d / vc (inf when not really approaching)
      near  the rule: (close AND moving relative to each other) OR (about to collide, a bit further out)
    Returns None when A and B share fewer than 2 frames.
    """
    ncfg = _rule(cfg, "near_miss")
    _, ia, ib = np.intersect1d(a.frame, b.frame, assume_unique=True, return_indices=True)
    if len(ia) < 2:
        return None
    vulnerable = {int(c) for c in ncfg.get("vulnerable_classes", [0])}
    both_people = b.cls in vulnerable
    t = a.t[ia]
    sa, sb = a.height[ia], b.height[ib]
    scale = 0.5 * (sa + sb) if both_people else sa
    scale = np.where(scale > 0, scale, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.hypot(*(a.foot[ia] - b.foot[ib]).T) / scale
        rel = np.hypot(*(a.vel[ia] - b.vel[ib]).T) / scale

        # Closing speed over the same time window used for speed (never across a tracking gap).
        window = float(cfg["features"]["speed_window_s"])
        segment = np.concatenate(([0], np.cumsum(np.diff(t) > float(cfg["features"]["max_gap_s"]))))
        lo = window_start(t, segment, window)
        dt = t - t[lo]
        vc = np.full(len(t), np.nan)
        ok = (dt >= 0.5 * window - 1e-9) & (dt > 0)
        vc[ok] = -(d[ok] - d[lo[ok]]) / dt[ok]
        ttc = np.where(vc > float(ncfg["min_closing_speed_bh_s"]), d / vc, np.inf)

        near_d = float(ncfg["near_distance_bh"])
        close_and_moving = (d <= near_d) & (rel >= float(ncfg["min_rel_speed_bh_s"]))
        about_to_hit = (ttc <= float(ncfg["ttc_s"])) & (d <= 1.5 * near_d)
        near = close_and_moving | about_to_hit
        if both_people:                    # very different sizes = different depths: only overlap in the image
            bigger = np.maximum(sa, sb)
            different_depth = np.abs(sa - sb) / np.where(bigger > 0, bigger, np.nan) > float(ncfg["max_scale_diff"])
            near &= ~different_depth
    return {"t": t, "ia": ia, "ib": ib, "segment": segment, "d": d, "rel": rel, "vc": vc, "ttc": ttc,
            "near": near}


def near_miss_intervals(tracks: dict[int, Track], cfg: dict) -> list[dict]:
    """Raw near-miss intervals: runs of "near" samples for each pair. A single-sample run is widened by
    +-0.25 s so it has a span. Each interval has track_id = A, entities [A, B] and other_id = B."""
    out: list[dict] = []
    max_gap = float(cfg["features"]["max_gap_s"])
    for a, b in near_miss_pairs(tracks, cfg):
        sig = pair_signals(a, b, cfg)
        if sig is None or not sig["near"].any():
            continue
        for s, e, i0, i1 in mask_to_intervals(sig["t"], sig["near"], max_gap):
            if i0 == i1:
                s, e = max(0.0, s - NEAR_PAD_S), e + NEAR_PAD_S
            out.append(_interval(a.track_id, "near_miss", s, e, entities=[a.track_id, b.track_id],
                                 other_id=b.track_id))
    return out


def near_miss_metrics(a: Track, b: Track, start_s: float, end_s: float, cfg: dict) -> dict:
    """Numbers for one (merged) near-miss span of the pair (A, B). See DESIGN 4.8 for the keys.

    min_distance_bh      closest the two feet came (in body-scale units)
    max_closing_speed    fastest approach inside the span
    min_ttc_s            shortest time-to-collision inside the span (None if never really approaching)
    approach_s           how long the distance had been falling before the closest moment
    contact              closest distance <= near_miss.contact_bh
    """
    ncfg = _rule(cfg, "near_miss")
    sig = pair_signals(a, b, cfg)
    out = {"min_distance_bh": None, "peak_time_s": round(0.5 * (start_s + end_s), 3),
           "max_closing_speed_bh_s": 0.0, "min_ttc_s": None, "rel_speed_at_closest_bh_s": None,
           "other_id": int(b.track_id), "other_class": int(b.cls), "approach_s": 0.0, "contact": False,
           "duration_s": round(max(0.0, end_s - start_s), 3), "mean_det_conf": 0.0,
           "min_duration_s": float(ncfg.get("min_duration_s", 0.0))}
    if sig is None:
        return out
    t, d = sig["t"], sig["d"]
    idx = _times_in_span(t, start_s, end_s)
    d_span = d[idx]
    if not np.isfinite(d_span).any():
        return out
    best = int(idx[int(np.nanargmin(d_span))])                       # index of the closest moment (whole series)
    vc, ttc, rel = sig["vc"][idx], sig["ttc"][idx], sig["rel"]
    closing = vc[np.isfinite(vc)]
    finite_ttc = ttc[np.isfinite(ttc)]
    # How long was the distance falling before the closest moment? Walk back while they were really
    # closing in (closing speed above the rule's minimum). A flat stretch (both standing still) is
    # NOT approaching, so it must not count.
    min_close = float(cfg["behaviors"]["near_miss"]["min_closing_speed_bh_s"])
    vc_all = sig["vc"]
    k = best
    while (k > 0 and sig["segment"][k - 1] == sig["segment"][k] and d[k - 1] >= d[k] - APPROACH_TOL_BH
           and np.isfinite(vc_all[k]) and vc_all[k] > min_close):
        k -= 1
    min_d = float(d[best])
    out.update({
        "min_distance_bh": round(min_d, 3),
        "peak_time_s": round(float(t[best]), 3),
        "max_closing_speed_bh_s": round(max(0.0, float(closing.max())), 3) if len(closing) else 0.0,
        "min_ttc_s": round(float(finite_ttc.min()), 2) if len(finite_ttc) else None,
        "rel_speed_at_closest_bh_s": round(float(rel[best]), 3) if math.isfinite(rel[best]) else None,
        "approach_s": round(float(t[best] - t[k]), 2),
        "contact": bool(min_d <= float(ncfg.get("contact_bh", 0.0))),
        "mean_det_conf": round(float(0.5 * (a.conf[sig["ia"][idx]] + b.conf[sig["ib"][idx]]).mean()), 3),
    })
    return out


# --------------------------------------------------------------------------- main entry point

def detect_behaviors(tracks: dict[int, Track], zones: list[dict], cfg: dict) -> dict:
    """Run every enabled rule on every track (and on pairs / the whole scene).

    Returns {"intervals": [...], "near_misses": [...], "baseline": {...}} as described in DESIGN 3.3.
    Zone intervals carry an extra "zone" key with the zone name. Crowding intervals have
    track_id None plus "zone" and "entities"; near-miss intervals have track_id A plus "entities"
    and "other_id".
    """
    bcfg = cfg["behaviors"]
    ratio = float(cfg["events"]["near_miss_ratio"])
    unit, short = unit_name(cfg), _short_unit(cfg)
    intervals: list[dict] = []
    near_misses: list[dict] = []
    dwell_cache: dict[int, float] = {}

    def dwell_of(track: Track) -> float:
        """Longest stationary stretch of the track (computed once per track)."""
        if track.track_id not in dwell_cache:
            dwell_cache[track.track_id] = stationary_runs(track, float(bcfg["loitering"]["radius_bh"]),
                                                          float(bcfg["loitering"].get("bridge_gap_s", 0.0)))
        return dwell_cache[track.track_id]

    for tid in sorted(tracks):
        track = tracks[tid]
        who = entity_name(cfg, tid, track.cls)

        # ---- loitering
        lcfg = bcfg["loitering"]
        if lcfg["enabled"]:
            found = loitering_intervals(track, cfg)
            intervals += [_interval(tid, "loitering", s, e) for s, e in found]
            if not found:
                dwell = dwell_of(track)
                if dwell >= ratio * float(lcfg["min_duration_s"]) - _EPS:
                    near_misses.append({
                        "track_id": tid, "behavior": "loitering", "value": round(dwell, 1),
                        "threshold": float(lcfg["min_duration_s"]), "unit": "s",
                        "note": f"{who} stood still for {dwell:.1f} s "
                                f"(rule: {float(lcfg['min_duration_s']):g} s). Not flagged."})

        # ---- zone intrusion
        zcfg = bcfg["zone_intrusion"]
        if zcfg["enabled"] and zones:
            found_z = zone_intervals(track, zones, cfg)
            intervals += [_interval(tid, "zone_intrusion", s, e, zone=name) for name, s, e in found_z]
            if found_z:
                # Was inside a zone, but never for long enough: a near miss (events.py may refine it
                # after merging the pieces).
                name, s, e = max(found_z, key=lambda z: z[2] - z[1])
                longest, need = e - s, float(zcfg["min_duration_s"])
                if longest + _EPS < need and longest >= ratio * need - _EPS:
                    near_misses.append({
                        "track_id": tid, "behavior": "zone_intrusion", "value": round(longest, 1),
                        "threshold": need, "unit": "s",
                        "note": f"{who} stepped into zone '{name}' for {longest:.1f} s "
                                f"(rule: >= {need:g} s). Not flagged."})

        # ---- running (samples where the object is down are ignored)
        rcfg = bcfg["running"]
        if rcfg["enabled"]:
            found = running_intervals(track, cfg)
            intervals += [_interval(tid, "running", s, e) for s, e in found]
            if not found:
                top, start_thr = _max_speed(track), float(rcfg["start_speed_bh_s"])
                if top >= ratio * start_thr - _EPS:
                    near_misses.append({
                        "track_id": tid, "behavior": "running", "value": round(top, 2),
                        "threshold": start_thr, "unit": f"{short}/s",
                        "note": f"{who} reached {top:.2f} {unit}/s "
                                f"(rule: >= {start_thr:g}). Not flagged."})

        # ---- fall / person down
        if fall_applies(track, cfg):
            intervals += [_interval(tid, "fall", s, e) for s, e in fall_intervals(track, cfg)]

        # ---- approaching (moving / wearable camera only; off in the fixed-camera presets)
        if approaching.applies(track, cfg):
            intervals += [_interval(tid, "approaching", s, e)
                          for s, e in approaching.approaching_intervals(track, cfg)]

    # ---- scene-level and pairwise rules need all tracks at once
    if _rule(cfg, "crowding").get("enabled"):
        intervals += crowding_intervals(tracks, zones, cfg)
    if _rule(cfg, "near_miss").get("enabled"):
        intervals += near_miss_intervals(tracks, cfg)

    baseline = compute_baseline(tracks, cfg, dwell_of)
    return {"intervals": intervals, "near_misses": near_misses, "baseline": baseline}


def _interval(track_id: int | None, behavior: str, start_s: float, end_s: float, zone: str | None = None,
              entities: list[int] | None = None, other_id: int | None = None) -> dict:
    """One raw interval dict. Optional keys appear only when they are used."""
    item = {"track_id": None if track_id is None else int(track_id), "behavior": behavior,
            "start_s": round(float(start_s), 3), "end_s": round(float(end_s), 3)}
    if zone is not None:
        item["zone"] = zone
    if entities is not None:
        item["entities"] = [int(e) for e in entities]
    if other_id is not None:
        item["other_id"] = int(other_id)
    return item


# --------------------------------------------------------------------------- baseline

def compute_baseline(tracks: dict[int, Track], cfg: dict, dwell_of) -> dict:
    """Scene-relative "normal": how does each object compare with the others of its kind in this video?

    A robust z-score (median / MAD, not mean / std, so one runner does not distort what "normal"
    is) is computed for each object's median speed and longest stillness. The anomaly score is the
    larger of the two (negative = calmer than usual, which is ignored). Needs at least
    baseline.min_tracks objects of the same class, otherwise that class is not judged (and when no
    class has enough, the baseline is inactive). In a one-class scene (the default) this is simply
    "all the people".
    """
    bl = cfg["behaviors"]["baseline"]
    eligible = [tid for tid in sorted(tracks)
                if tracks[tid].duration_s >= BASELINE_MIN_TRACK_S and math.isfinite(_median_speed(tracks[tid]))]
    result = {"enabled": bool(bl["enabled"]), "active": False, "n_tracks": len(eligible),
              "median_speed_bh_s": 0.0, "median_dwell_s": 0.0, "tracks": {}}
    if not bl["enabled"] or len(eligible) == 0:
        return result

    # Compare like with like: people with people, cars with cars.
    groups: dict[int, list[int]] = {}
    for tid in eligible:
        groups.setdefault(tracks[tid].cls, []).append(tid)
    biggest = max(groups.values(), key=len)
    speeds = np.array([_median_speed(tracks[tid]) for tid in biggest])
    dwells = np.array([dwell_of(tracks[tid]) for tid in biggest])
    result["median_speed_bh_s"] = round(float(np.median(speeds)), 3)
    result["median_dwell_s"] = round(float(np.median(dwells)), 2)

    z_thr = float(bl["z_threshold"])
    for cls, members in groups.items():
        if len(members) < int(bl["min_tracks"]):
            continue                                        # too few of this kind to define "normal"
        speeds = np.array([_median_speed(tracks[tid]) for tid in members])
        dwells = np.array([dwell_of(tracks[tid]) for tid in members])
        speed_z, med_speed, _ = robust_z(speeds)
        dwell_z, med_dwell, _ = robust_z(dwells)
        result["active"] = True
        for k, tid in enumerate(members):
            score = float(max(speed_z[k], dwell_z[k], 0.0))
            unusual = score >= z_thr
            result["tracks"][tid] = {
                "median_speed_bh_s": round(float(speeds[k]), 3),
                "max_dwell_s": round(float(dwells[k]), 2),
                "speed_z": round(float(speed_z[k]), 2),
                "dwell_z": round(float(dwell_z[k]), 2),
                "anomaly_score": round(score, 2),
                "unusual": bool(unusual),
                "note": _baseline_note(unusual, float(speeds[k]), float(dwells[k]), float(speed_z[k]),
                                       float(dwell_z[k]), med_speed, med_dwell, unit_name(cfg),
                                       _typical_word(cfg, cls)),
            }
    result["tracks"] = dict(sorted(result["tracks"].items()))
    return result


def _typical_word(cfg: dict, cls: int) -> str:
    """"person" / "animal" / "car": what the baseline note compares the object with."""
    classes = ((cfg.get("model") or {}).get("classes")) or [0]
    if len(classes) > 1 and not state_classes(cfg):          # state detector: one kind of entity
        return CLASS_NAMES.get(int(cls), "object").lower()
    return entity_word(cfg).lower()


def _baseline_note(unusual: bool, speed: float, dwell: float, speed_z: float, dwell_z: float,
                   med_speed: float, med_dwell: float, unit: str = "body-heights",
                   word: str = "person") -> str:
    """Plain-English comparison with the scene's typical object (ratios to the scene median)."""
    if not unusual:
        return "Within the normal range for this scene"
    if speed_z >= dwell_z:                                   # the speed is what stands out
        if med_speed > 0.05:
            return (f"Moved {speed / med_speed:.1f}x faster than the scene's typical {word} "
                    f"({speed:.2f} vs {med_speed:.2f} {unit}/s)")
        return f"Moved at {speed:.2f} {unit}/s while the scene's typical {word} is almost still"
    if med_dwell > 0.05:                                     # the stillness is what stands out
        return (f"Stayed near one spot {dwell / med_dwell:.1f}x longer than the scene's typical {word} "
                f"({dwell:.0f} s vs {med_dwell:.1f} s)")
    return f"Stayed near one spot for {dwell:.0f} s while the scene's typical {word} keeps moving"


# --------------------------------------------------------------------------- metrics for final events

def compute_metrics(track: Track | None, behavior: str, start_s: float, end_s: float, zones: list[dict],
                    cfg: dict, zone_name: str | None = None, other: Track | None = None,
                    tracks: dict[int, Track] | None = None) -> dict:
    """Numbers that justify an event, measured on the (merged) span [start_s, end_s].

    `zone_name` is an optional extra argument: events.py passes the zone the raw intervals belonged
    to; if it is omitted the zone that contains most of the span's samples is used.
    Two behaviours need more than one track: pass `tracks` (all of them) for "crowding" (then `track`
    may be None), and `other` (the second track B) for "near_miss" (`track` is A).
    """
    if behavior not in BEHAVIORS:
        raise ValueError(f"unknown behavior: {behavior!r} (expected one of {BEHAVIORS})")
    if behavior == "crowding":
        if tracks is None:
            raise ValueError("compute_metrics('crowding') needs tracks=<all tracks>")
        return crowding_metrics(tracks, zones, start_s, end_s, cfg, zone_name)[0]
    if behavior == "near_miss":
        if track is None or other is None:
            raise ValueError("compute_metrics('near_miss') needs track=A and other=B")
        return near_miss_metrics(track, other, start_s, end_s, cfg)
    if behavior == "approaching":
        return approaching.approaching_metrics(track, start_s, end_s, cfg)

    idx = _span_indices(track, start_s, end_s)
    foot, height, speed = track.foot[idx], track.height[idx], track.speed[idx]
    finite_speed = speed[np.isfinite(speed)]
    bcfg = cfg["behaviors"][behavior]
    mid = 0.5 * (start_s + end_s)
    out: dict = {"duration_s": round(max(0.0, end_s - start_s), 3)}

    if behavior == "loitering":
        out["max_radius_bh"] = round(window_radius_bh(foot, height), 3)
        # Extra (not in DESIGN): the loitering span includes the walk-in / walk-out steps that still fit
        # inside the radius, which pushes max_radius_bh towards the limit even for a person who stood
        # perfectly still. p90 = "90% of the time within this distance of the spot" ignores those
        # short tails, so events.py uses it for the confidence margin.
        out["p90_radius_bh"] = round(_p90_radius_bh(foot, height), 3)
        out["radius_threshold_bh"] = float(bcfg["radius_bh"])
        out["min_duration_s"] = float(bcfg["min_duration_s"])
        out["mean_speed_bh_s"] = round(float(finite_speed.mean()), 3) if len(finite_speed) else 0.0
        peak_time = mid
    elif behavior == "zone_intrusion":
        name, depth = _zone_depth(track, idx, zones, zone_name)
        out["zone"] = name
        out["max_depth_bh"] = round(float(depth.max()), 3) if len(depth) else 0.0
        out["min_duration_s"] = float(bcfg["min_duration_s"])
        peak_time = float(track.t[idx][int(np.argmax(depth))]) if len(depth) and depth.max() > 0 else mid
    elif behavior == "fall":
        aspect = track.aspect[idx]
        out["max_aspect"] = round(float(np.nanmax(aspect)), 3) if np.isfinite(aspect).any() else 0.0
        upright_before = _upright_aspect_before(track, start_s, float(bcfg["transition_s"]))
        out["upright_aspect_before"] = None if upright_before is None else round(upright_before, 3)
        out["fell"] = bool(upright_before is not None and upright_before <= float(bcfg["upright_max_aspect"]))
        out["transition_s"] = float(bcfg["transition_s"])
        out["min_down_s"] = float(bcfg["min_down_s"])
        peak_time = min(start_s + FALL_PEAK_OFFSET_S, end_s)
    else:  # running
        out["peak_speed_bh_s"] = round(float(finite_speed.max()), 3) if len(finite_speed) else 0.0
        out["mean_speed_bh_s"] = round(float(finite_speed.mean()), 3) if len(finite_speed) else 0.0
        out["start_threshold_bh_s"] = float(bcfg["start_speed_bh_s"])
        out["end_threshold_bh_s"] = float(bcfg["end_speed_bh_s"])
        out["min_duration_s"] = float(bcfg["min_duration_s"])
        peak_time = float(track.t[idx][int(np.nanargmax(speed))]) if len(finite_speed) else mid

    out["mean_det_conf"] = round(float(track.conf[idx].mean()), 3)
    out["peak_time_s"] = round(float(peak_time), 3)
    return out


def _upright_aspect_before(track: Track, start_s: float, transition_s: float) -> float | None:
    """Lowest box aspect (w/h) in the `transition_s` seconds before `start_s`; None if nothing was seen then.

    A low value (<= fall.upright_max_aspect) means the person was standing just before going down.
    """
    before = (track.t >= start_s - transition_s - _EPS) & (track.t < start_s - _EPS)
    values = track.aspect[before]
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else None


def _p90_radius_bh(foot: np.ndarray, height: np.ndarray) -> float:
    """90th-percentile distance from the span's median position, in body-scale units (robust spread)."""
    centre = np.median(foot, axis=0)
    dist = np.hypot(foot[:, 0] - centre[0], foot[:, 1] - centre[1])
    bh = float(np.median(height))
    return float(np.percentile(dist, 90) / bh) if bh > 0 else float("inf")


def _zone_depth(track: Track, idx: np.ndarray, zones: list[dict], zone_name: str | None):
    """(zone name, depth per sample in body-scale units) for the span; depth is 0 for samples outside.

    Depth = distance from the foot point to the nearest polygon edge / body scale, so a person
    standing right on the border is shallow (about 0) and one deep inside is a larger number.
    """
    named = [(_zone_name(z, k), z["points"]) for k, z in enumerate(zones)]
    if not named:
        return zone_name, np.zeros(0)
    if zone_name is None:                                  # infer: the zone with most samples inside
        counts = [sum(point_in_polygon(float(track.foot[i, 0]), float(track.foot[i, 1]), pts) for i in idx)
                  for _, pts in named]
        zone_name = named[int(np.argmax(counts))][0]
    pts = next((p for n, p in named if n == zone_name), named[0][1])
    depth = np.zeros(len(idx))
    for k, i in enumerate(idx):
        x, y = float(track.foot[i, 0]), float(track.foot[i, 1])
        if point_in_polygon(x, y, pts) and track.height[i] > 0:
            depth[k] = distance_to_polygon_edge(x, y, pts) / float(track.height[i])
    return zone_name, depth
