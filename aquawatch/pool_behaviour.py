"""Stage 2 of the pool pipeline: what is each swimmer DOING right now? (behaviour signals)

Every `pool.sample_s` seconds, for every tracked swimmer, we look at the last `pool.window_s` seconds
and describe the behaviour with plain signals (the team's stage-2 data contract, plus extra fields):

    {"person_id": 3, "timestamp": 84.2, "posture": "vertical", "movement": "low",
     "arm_motion": "repeated", "displacement": 0.05, "head": "unstable",
     "near_wall": false, "in_water": true, "location": "Deep End",
     "progress_ratio": 0.03, "arm_hz": 1.3, "arm_regularity": 0.12, "head_visibility": 0.9,
     "torso_angle": null, "activity": "floundering", "confidence": 0.86, "track_valid": true, ...}

* posture       horizontal / diagonal / vertical / head_only / underwater / out_of_water. Source, in order:
                the custom swimmer detector's class (pool.state_classes), else the shoulder-hip TORSO ANGLE
                when the hips are visible, else the box shape (in water the hips are usually hidden).
                A new posture must last pool.posture_hold_s before it is reported (no flicker).
* movement      low / medium / high: path speed of the box centre in body-heights per second.
* displacement  NET progress in body-heights per second (treading water in place ~ 0).
  progress_ratio  net / path length (1 = straight line, ~0 = moving but going nowhere).
* arm_motion    repeated / stroking / calm / irregular: wrist peaks per second relative to the shoulders
                (pose keypoints), or the top of the box bobbing when there is no pose (arm_source says which).
                "stroking" = rhythmic arms while really travelling (swimming strokes, not distress).
* head          stable / unstable / submerged / unknown: head visibility, bobbing, and (when upright)
                whether the head is held clearly above the shoulders.
* activity      a HINT for the dashboard and debugging (swimming / floating / treading / floundering /
                submerged / unclear). Stage 3 (distress.py) never uses it as a decision.
* track_valid   false for pool.window_s after the box jumped (ID switch suspected): stage 3 ignores it.

Several ideas (torso angle, posture hold, progress ratio, peak-counted arm rate and regularity, head
visibility, ID-switch guard, keypoint smoothing, confidence, activity hint, signal plots) come from the
team's behaviour_analysis.py prototype, merged into this pipeline's data contract.

In water the box CENTRE is used (feet are under water). Distances are divided by the swimmer's usual
box size, so the same numbers work near and far from the camera. NumPy only. Run on its own:
    python pool_behaviour.py --tracks outputs/x/stage1_tracking.jsonl --zones zones.json --out stage2.jsonl --plots
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import utils

POSTURES = ("horizontal", "diagonal", "vertical", "head_only", "underwater", "out_of_water")
STATE_TO_POSTURE = {"swimmer_horizontal": "horizontal", "swimmer_vertical": "vertical", "head_only": "head_only",
                    "underwater": "underwater", "out_of_water": "out_of_water", "person_out_of_water": "out_of_water"}
# COCO keypoints used here
NOSE, L_EYE, R_EYE, L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP = 0, 1, 2, 5, 6, 9, 10, 11, 12


# --------------------------------------------------------------------------- zones

def _zone_kind(name: str, pcfg: dict) -> str:
    """'edge' (next to the wall), 'water' (the water area) or 'location' (Deep End, Shallow End ...)."""
    low = name.lower()
    if any(w in low for w in pcfg.get("edge_zone_words", [])):
        return "edge"
    if any(w in low for w in pcfg.get("water_zone_words", [])):
        return "water"
    return "location"


def place_of(x: float, y: float, zones: list[dict], pcfg: dict) -> dict:
    """Where a point is: near_wall, in_water (True when no water zone is drawn) and the location name."""
    near_wall, location = False, None
    water_zones = [z for z in zones if _zone_kind(z["name"], pcfg) == "water"]
    in_water = not water_zones or any(utils.point_in_polygon(x, y, z["points"]) for z in water_zones)
    for z in zones:
        if not utils.point_in_polygon(x, y, z["points"]):
            continue
        kind = _zone_kind(z["name"], pcfg)
        if kind == "edge":
            near_wall = True
        elif kind == "location" and location is None:
            location = z["name"]
    return {"near_wall": near_wall, "in_water": bool(in_water),
            "location": location or pcfg.get("default_location", "Pool")}


# --------------------------------------------------------------------------- signal helpers

def find_peaks(x, prominence: float = 0.0) -> np.ndarray:
    """Indices of local maxima whose topographic prominence is >= `prominence` (like scipy.signal.find_peaks)."""
    x = np.asarray(x, dtype=float)
    n, peaks = len(x), []
    for i in range(1, n - 1):
        if not (x[i] > x[i - 1] and x[i] >= x[i + 1]):
            continue
        lo, lmin = i, x[i]
        while lo > 0 and x[lo - 1] <= x[i]:
            lo -= 1
            lmin = min(lmin, x[lo])
        hi, rmin = i, x[i]
        while hi < n - 1 and x[hi + 1] <= x[i]:
            hi += 1
            rmin = min(rmin, x[hi])
        if x[i] - max(lmin, rmin) >= prominence:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _rhythm(signal: np.ndarray, t: np.ndarray, prominence: float) -> tuple[float, float, float | None]:
    """(amplitude, peaks per second, regularity) of an up-and-down signal.

    amplitude = std after removing the slow drift; regularity = std/mean of the peak intervals
    (small = rhythmic, None when fewer than 3 peaks).
    """
    if len(signal) < 4 or t[-1] - t[0] <= 0:
        return 0.0, 0.0, None
    x = signal - np.polyval(np.polyfit(t, signal, 1), t)      # remove slow drift (sinking, moving away)
    amp = float(np.std(x))
    pk = find_peaks(x, prominence)
    rate = len(pk) / float(t[-1] - t[0])
    reg = None
    if len(pk) >= 3:
        iv = np.diff(t[pk])
        reg = float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else None
    return amp, rate, reg


def posture_from_box(box: np.ndarray, usual_area: float, pcfg: dict) -> str:
    """Stock-detector fallback: long and flat = horizontal; much smaller than usual = head only; else vertical."""
    w, h = float(box[2] - box[0]), float(box[3] - box[1])
    if h <= 0 or w <= 0:
        return "vertical"
    if w / h >= float(pcfg["horizontal_aspect"]):
        return "horizontal"
    if usual_area > 0 and (w * h) / usual_area <= float(pcfg["head_area_frac"]):
        return "head_only"
    return "vertical"


def torso_angle(kps, kp_conf: float):
    """Angle of the shoulder->hip line from horizontal in degrees (90 = upright), or None if hips/shoulders hidden."""
    k = np.asarray(kps, dtype=float)
    if k.shape != (17, 3) or min(k[L_SH, 2], k[R_SH, 2], k[L_HIP, 2], k[R_HIP, 2]) < kp_conf:
        return None
    sh, hp = (k[L_SH, :2] + k[R_SH, :2]) / 2, (k[L_HIP, :2] + k[R_HIP, :2]) / 2
    return math.degrees(math.atan2(abs(sh[1] - hp[1]), abs(sh[0] - hp[0]) + 1e-9))


def posture_from_angle(angle: float, pcfg: dict) -> str:
    """vertical / horizontal / diagonal from the torso angle."""
    if angle >= float(pcfg["vertical_above_deg"]):
        return "vertical"
    if angle <= float(pcfg["horizontal_below_deg"]):
        return "horizontal"
    return "diagonal"


def arm_signal(kps, scale: float, kp_conf: float):
    """Mean wrist height relative to the shoulders (body-heights; positive = wrist above shoulder), or None."""
    k = np.asarray(kps, dtype=float)
    if k.shape != (17, 3):
        return None
    shoulders = [k[j] for j in (L_SH, R_SH) if k[j, 2] >= kp_conf]
    wrists = [k[j] for j in (L_WR, R_WR) if k[j, 2] >= kp_conf]
    if not shoulders or not wrists or scale <= 0:
        return None
    sy = float(np.mean([s[1] for s in shoulders]))
    return float(np.mean([(sy - w[1]) / scale for w in wrists]))


def head_signal(kps, scale: float, kp_conf: float):
    """(head height in body-heights from the frame top, head height above the shoulders) or None if unseen."""
    k = np.asarray(kps, dtype=float)
    if k.shape != (17, 3) or scale <= 0:
        return None
    pts = [k[j] for j in (NOSE, L_EYE, R_EYE) if k[j, 2] >= kp_conf]
    if not pts:
        return None
    hy = float(np.mean([p[1] for p in pts]))
    sh = [k[j] for j in (L_SH, R_SH) if k[j, 2] >= kp_conf]
    above = (float(np.mean([s[1] for s in sh])) - hy) / scale if sh else None
    return hy / scale, above


def smooth_keypoints(kp_track: dict, kp_conf: float, alpha: float, max_gap: int) -> dict:
    """Exponential smoothing of each keypoint over time (pose jitter), holding a missing keypoint for up
    to `max_gap` samples before dropping it. {frame: 17x3} -> {frame: 17x3} (confidence kept)."""
    out, sm, age = {}, None, np.full(17, 99)
    for f in sorted(kp_track):
        k = np.asarray(kp_track[f], dtype=float)
        if k.shape != (17, 3):
            continue
        valid = k[:, 2] >= kp_conf
        if sm is None:
            sm = k.copy()
            age = np.where(valid, 0, 99)
        else:
            for i in range(17):
                if valid[i]:
                    if age[i] > max_gap or sm[i, 2] < kp_conf:
                        sm[i] = k[i]
                    else:
                        sm[i, :2] = alpha * k[i, :2] + (1 - alpha) * sm[i, :2]
                        sm[i, 2] = k[i, 2]
                    age[i] = 0
                else:
                    age[i] += 1
                    if age[i] > max_gap:
                        sm[i, 2] = 0.0                    # gone for too long: treat as missing
        out[f] = sm.copy().tolist()
    return out


def activity_hint(posture: str, net_low: bool, travelling: bool, arm: str, head: str) -> str:
    """Plain label for the dashboard / debugging (never a decision): swimming, floating, treading,
    floundering, submerged or unclear."""
    if head == "submerged":
        return "submerged"
    upright = posture in ("vertical", "head_only", "diagonal")
    if upright and net_low and arm == "repeated":
        return "floundering"
    if upright and net_low and arm == "calm" and head == "stable":
        return "treading"
    if posture in ("horizontal", "diagonal") and travelling:
        return "swimming"
    if posture == "horizontal" and net_low and arm == "calm":
        return "floating"
    return "unclear"


# --------------------------------------------------------------------------- the stage

def _piece_index(t: np.ndarray, spans) -> np.ndarray:
    """Which joined piece (stitch.py) each detection time belongs to: 0 everywhere for a track that was not joined."""
    if not spans:
        return np.zeros(len(t), dtype=int)
    starts = np.array(sorted(float(a) for a, _ in spans))
    return np.clip(np.searchsorted(starts, t + 1e-9, side="right") - 1, 0, len(starts) - 1)


def behaviour_samples(tracks: dict, cfg: dict, zones: list[dict] | None = None,
                      keypoints: dict | None = None, det_classes: dict | None = None,
                      pieces: dict | None = None) -> list[dict]:
    """Stage-2 samples for every swimmer, sorted by (person_id, timestamp).

    `keypoints`: {track_id: {frame_idx: 17x3 list}} from the pose model (processing pixels), optional.
    `det_classes`: {(track_id, frame_idx): class_id} from the detector, used with pool.state_classes.
    `pieces`: {track_id: [(start, end), ...]} of a track joined by stitch.py. Each piece keeps its OWN body size and
    usual box size, so joining a far piece with a near one (camera zoom, swimmer coming closer) never changes how a
    piece's motion is measured.
    """
    pcfg = cfg["pool"]
    zones = zones or []
    keypoints = keypoints or {}
    det_classes = det_classes or {}
    state_map = {int(k): str(v) for k, v in (pcfg.get("state_classes") or {}).items()}
    step, window = float(pcfg["sample_s"]), float(pcfg["window_s"])
    kp_conf = float(pcfg["kp_conf"])
    out = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        t = np.asarray(tr.t, dtype=float)
        if len(t) < 3:
            continue
        box = np.asarray(tr.box, dtype=float).reshape(-1, 4)
        centre = np.column_stack(((box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2))
        jumps = _jump_times(t, centre, box, pcfg)              # ID switch suspected at these times
        centre = utils.moving_average(centre, t, float(cfg["features"]["smoothing_s"]))
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        height = np.asarray(tr.height, dtype=float)
        piece_of = _piece_index(t, (pieces or {}).get(tid))      # all 0 when the track was not joined
        piece_area = {int(p): float(np.median(area[piece_of == p])) for p in np.unique(piece_of)}
        piece_scale = {int(p): float(np.median(height[piece_of == p])) or 1.0 for p in np.unique(piece_of)}
        raw_kp = {int(f): v for f, v in (keypoints.get(tid) or keypoints.get(str(tid)) or {}).items()}
        kp_track = smooth_keypoints(raw_kp, kp_conf, float(pcfg["ema_alpha"]), int(pcfg["kp_max_gap"]))

        records = []

        def sample_at(ts):
            idx = np.flatnonzero((t >= ts - window - 1e-9) & (t <= ts + 1e-9))
            if len(idx) >= 3 and t[idx[-1]] - t[idx[0]] >= 0.5 * window:
                p = int(piece_of[idx[-1]])                          # the piece this moment belongs to
                rec = _one_sample(tid, ts, idx, tr, t, box, centre, area, piece_area[p], piece_scale[p],
                                  kp_track, raw_kp, det_classes, state_map, zones, pcfg, kp_conf)
                rec["track_valid"] = not any(ts - window < j <= ts for j in jumps)
                records.append(rec)

        ts = t[0] + window
        while ts <= t[-1] + 1e-9:
            sample_at(ts)
            ts += step
        # the clock never lands on the last detection: one more window ending there, so the last moments (and a
        # track shorter than window_s) are judged too (bug hunt 2026-10-11, pool low; bug group B step B7)
        if not records or records[-1]["timestamp"] < float(t[-1]) - 0.5 * step:
            sample_at(float(t[-1]))
        _hold_posture(records, float(pcfg["posture_hold_s"]), jumps)
        for rec in records:                                 # the hint uses the HELD posture
            rec["activity"] = activity_hint(rec["posture"], rec["displacement"] <= float(pcfg["displacement_low_bh_s"]),
                                            rec["progress_ratio"] >= 0.5 and rec["movement"] != "low",
                                            rec["arm_motion"], rec["head"])
        out += records
    return out


def _jump_times(t, centre, box, pcfg) -> list[float]:
    """Times where the box centre jumped more than jump_factor x its size within 1 s (ID switch suspected)."""
    jumps = []
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 1.0:
            continue
        size = max(box[i - 1, 2] - box[i - 1, 0], box[i - 1, 3] - box[i - 1, 1], 1.0)
        if np.linalg.norm(centre[i] - centre[i - 1]) > float(pcfg["jump_factor"]) * size:
            jumps.append(float(t[i]))
    return jumps


def _hold_posture(records: list[dict], hold_s: float, jumps: list[float]) -> None:
    """A new posture is only reported once it has lasted hold_s (raw value kept in posture_raw)."""
    current, pending, since = None, None, None
    for r in records:
        raw = r["posture_raw"] = r["posture"]
        if any(r["timestamp"] - 1e-9 <= j <= r["timestamp"] + 1e-9 for j in jumps):
            current, pending = None, None                    # new identity: start over
        if current is None or raw == current:
            current, pending = raw, None
        elif raw != pending:
            pending, since = raw, r["timestamp"]
        elif r["timestamp"] - since >= hold_s - 1e-9:
            current, pending = raw, None
        r["posture"] = current


def _majority(votes: list[str]) -> str:
    """Most frequent vote; a tie goes to the most recent of the tied values (deterministic, unlike max(set(...)),
    whose order changes between runs because string hashing is randomised)."""
    counts = {v: votes.count(v) for v in votes}
    top = max(counts.values())
    return next(v for v in reversed(votes) if counts[v] == top)


def _one_sample(tid, ts, idx, tr, t, box, centre, area, usual_area, scale, kp_track, raw_kp, det_classes,
                state_map, zones, pcfg, kp_conf) -> dict:
    """One stage-2 record for swimmer `tid` at time `ts` (window samples `idx`).

    Smoothed keypoints (kp_track) feed the slow signals (torso angle, head height); the RAW keypoints feed
    the rhythm signals (arm peaks, head bobbing), because smoothing would damp the very oscillation we look for.
    """
    dt = float(t[idx[-1]] - t[idx[0]]) or 1e-6
    steps = np.linalg.norm(np.diff(centre[idx], axis=0), axis=1)
    path = float(steps.sum() / scale)
    net = float(np.linalg.norm(centre[idx[-1]] - centre[idx[0]]) / scale)
    speed, displacement = path / dt, net / dt
    progress = net / path if path > 1e-6 else 0.0
    movement = ("low" if speed < float(pcfg["movement_low_bh_s"])
                else "high" if speed > float(pcfg["movement_high_bh_s"]) else "medium")

    frames = [int(f) for f in tr.frame[idx]]
    kps = [(f, tt, kp_track[f]) for f, tt in zip(frames, t[idx]) if f in kp_track]
    raws = [(f, tt, raw_kp[f]) for f, tt in zip(frames, t[idx]) if f in raw_kp]

    # posture: detector class > torso angle (hips visible) > box shape; majority over the window's second half
    recent = idx[len(idx) // 2:]
    votes, sources, angles = [], [], []
    for i in recent:
        f = int(tr.frame[i])
        cls = det_classes.get((tid, f))
        state = state_map.get(int(cls)) if cls is not None else None
        ang = torso_angle(kp_track[f], kp_conf) if f in kp_track else None
        if ang is not None:
            angles.append(ang)
        if state:
            p = STATE_TO_POSTURE.get(state, "vertical")
            # the trained model has no head_only examples yet: an upright box that shrank to a head is still head_only
            if p == "vertical" and posture_from_box(box[i], usual_area, pcfg) == "head_only":
                p = "head_only"
            votes.append(p)
            sources.append("detector")
        elif ang is not None:
            votes.append(posture_from_angle(ang, pcfg))
            sources.append("torso_angle")
        else:
            votes.append(posture_from_box(box[i], usual_area, pcfg))
            sources.append("box_shape")
    posture = _majority(votes)
    state_source = _majority(sources)

    # arms: wrist peaks relative to the shoulders (pose), else the top of the box bobbing
    arm_vals = [(tt, a) for _, tt, k in raws if (a := arm_signal(k, scale, kp_conf)) is not None]
    if len(arm_vals) >= 4:
        amp, rate, reg = _rhythm(np.array([v for _, v in arm_vals]), np.array([tt for tt, _ in arm_vals]),
                                 float(pcfg["arm_peak_prominence"]))
        arm_source = "pose"
    else:
        amp, rate, reg = _rhythm(box[idx, 1] / scale, t[idx], float(pcfg["arm_peak_prominence"]))
        arm_source = "box"
    travelling = progress >= 0.5 and displacement > float(pcfg["displacement_low_bh_s"])
    if amp >= float(pcfg["arm_min_amp"]) and rate >= float(pcfg["arm_min_hz"]):
        arm_motion = "stroking" if (travelling and posture in ("horizontal", "diagonal")) else "repeated"
    elif amp <= float(pcfg["arm_calm_amp"]) or rate < float(pcfg["arm_none_hz"]):
        arm_motion = "calm"
    else:
        arm_motion = "irregular"

    # head: visibility, bobbing and (when upright) height above the shoulders
    heads = [(tt, h) for _, tt, k in kps if (h := head_signal(k, scale, kp_conf)) is not None]
    bob = [(tt, h[0]) for _, tt, k in raws if (h := head_signal(k, scale, kp_conf)) is not None]
    visibility = len(bob) / len(raws) if raws else None
    shrinking = float(area[idx[-1]]) < float(pcfg["shrink_ratio"]) * float(np.max(area[idx]))
    head_bob, head_low = None, False                            # kept for the wave-relative check (wave_relative_heads)
    if len(heads) >= 3:
        head_amp, _, _ = _rhythm(np.array([v for _, v in bob]), np.array([tt for tt, _ in bob]), 1e9) \
            if len(bob) >= 3 else (0.0, 0.0, None)
        aboves = [h[1] for _, h in heads if h[1] is not None]
        low_head = bool(posture in ("vertical", "head_only") and aboves
                        and float(np.median(aboves)) < float(pcfg["head_up_min"]))
        head = "unstable" if (head_amp > float(pcfg["head_unstable"]) or low_head) else "stable"
        head_bob, head_low = float(head_amp), low_head
    elif kps and len(kps) >= 3 and visibility is not None and visibility < float(pcfg["head_submerged_below"]) \
            and (shrinking or posture in ("head_only", "underwater")):
        head = "submerged"
    elif kps and len(kps) >= 3:
        head = "unknown"
    else:                                                       # no pose: box top jitter as a weak proxy
        top_amp, _, _ = _rhythm(box[idx, 1] / scale, t[idx], 1e9)
        # the box-top bobbing is the same number the box 'arms' reading used: count it once (bug hunt 2026-10-11,
        # pool high; PREREGISTRATION_bug_group_B.md step B2)
        head = "submerged" if (shrinking and posture in ("head_only", "underwater")) else \
               ("unstable" if top_amp > float(pcfg["head_unstable"]) and arm_motion != "repeated" else "stable")
        head_bob = float(top_amp)
    if posture == "underwater":
        head = "submerged"

    # confidence: how much of the body the pose saw, how big the swimmer is, and whether we had to guess posture
    if kps:
        seen = np.mean([np.mean(np.asarray(k)[[NOSE, L_SH, R_SH, L_WR, R_WR], 2] >= kp_conf) for _, _, k in kps])
    else:
        seen = 0.5
    bw, bh = float(box[idx[-1], 2] - box[idx[-1], 0]), float(box[idx[-1], 3] - box[idx[-1], 1])
    size = min(1.0, max(bw, bh) / float(pcfg["min_box_px"]))
    conf = float(seen * size * (0.8 if state_source == "box_shape" else 1.0))

    place = place_of(float(centre[idx[-1], 0]), float(centre[idx[-1], 1]), zones, pcfg)
    x1, y1, x2, y2 = (float(v) for v in box[idx[-1]])
    return {
        "person_id": int(tid), "timestamp": round(float(ts), 2),
        "posture": posture, "movement": movement, "arm_motion": arm_motion,
        "displacement": round(displacement, 3), "speed": round(speed, 3), "head": head,
        "near_wall": place["near_wall"], "in_water": place["in_water"], "location": place["location"],
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "state_source": state_source, "arm_source": arm_source,
        "arm_amplitude": round(amp, 3), "arm_hz": round(rate, 2),
        "arm_regularity": None if reg is None else round(reg, 2),
        "torso_angle": round(float(np.median(angles)), 1) if angles else None,
        "net_displacement": round(net, 3), "path_length": round(path, 3), "progress_ratio": round(progress, 3),
        "head_visibility": None if visibility is None else round(visibility, 2),
        "head_bob": None if head_bob is None else round(head_bob, 4), "head_low": head_low,
        "confidence": round(conf, 2),
    }


def wave_relative_heads(samples: list[dict], pcfg: dict) -> list[dict]:
    """Rule B (pre-registered, data/ground_truth/PREREGISTRATION_vanish_waves.md): in a wave pool every head bobs,
    so an 'unstable' head from bobbing alone counts only when this swimmer bobs more than the swimmers around them:
    own bobbing minus the median bobbing of the others within head_neighbour_bh box heights, at the same moment,
    must exceed head_unstable. Fewer than head_min_neighbours neighbours, or a head low above the shoulders: as
    before. Changes the samples in place (head -> "stable", plus head_wave = the neighbours' median) and returns them."""
    if not pcfg.get("head_relative_to_neighbours"):
        return samples
    limit, radius = float(pcfg["head_unstable"]), float(pcfg["head_neighbour_bh"])
    need = int(pcfg["head_min_neighbours"])
    half_step = float(pcfg["sample_s"]) / 2          # every swimmer is sampled on its own clock: "same moment" =
    timed = sorted(samples, key=lambda s: float(s["timestamp"]))     # within half a sample step

    def centre_scale(s: dict) -> tuple[float, float, float]:
        x1, y1, x2, y2 = s["bbox"]
        return (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 1e-6)

    bobs = [o.get("head_bob") for o in timed]        # before any change: every swimmer is judged on the same data
    for s in timed:
        if s["head"] != "unstable" or s.get("head_low") or s.get("head_bob") is None:
            continue
        t, (cx, cy, scale) = float(s["timestamp"]), centre_scale(s)
        # one reading per neighbouring SWIMMER (the one nearest in time): two samples of one neighbour 0.25 s either
        # side must not count as two neighbours (bug hunt 2026-10-11, pool low; bug group B step B8)
        nearest: dict = {}
        for o, b in zip(timed, bobs):
            dt = abs(float(o["timestamp"]) - t)
            if b is None or o["person_id"] == s["person_id"] or dt > half_step                     or np.hypot(centre_scale(o)[0] - cx, centre_scale(o)[1] - cy) > radius * scale:
                continue
            if o["person_id"] not in nearest or dt < nearest[o["person_id"]][0]:
                nearest[o["person_id"]] = (dt, b)
        others = [b for _, b in nearest.values()]
        if len(others) < need:
            continue
        wave = float(np.median(others))
        if float(s["head_bob"]) - wave <= limit:
            s["head"], s["head_wave"] = "stable", round(wave, 4)
    return samples


# --------------------------------------------------------------------------- plots (debugging)

def plot_signals(samples: list[dict], out_dir, max_people: int = 6) -> list[str]:
    """One figure per swimmer with the behaviour signals over time (for tuning thresholds)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by: dict[int, list] = {}
    for s in samples:
        by.setdefault(s["person_id"], []).append(s)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pid in sorted(by, key=lambda k: -len(by[k]))[:max_people]:
        rs = sorted(by[pid], key=lambda r: r["timestamp"])
        t = [r["timestamp"] for r in rs]
        fig, ax = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
        ax[0].step(t, [POSTURES.index(r["posture"]) for r in rs], where="post")
        ax[0].set_yticks(range(len(POSTURES)), POSTURES, fontsize=7)
        ax[0].set_ylabel("posture")
        ax[1].plot(t, [r["displacement"] for r in rs])
        ax[1].set_ylabel("displacement\n(bh/s)")
        ax[2].plot(t, [r["progress_ratio"] for r in rs])
        ax[2].set_ylabel("progress ratio")
        ax[3].plot(t, [r["arm_hz"] for r in rs])
        ax[3].set_ylabel("arm peaks/s")
        ax[4].plot(t, [r["head_visibility"] if r["head_visibility"] is not None else np.nan for r in rs])
        ax[4].set_ylabel("head visibility")
        ax[4].set_xlabel("time (s)")
        fig.suptitle(f"Swimmer #{pid}: behaviour signals")
        fig.tight_layout()
        path = out_dir / f"person_{pid}_signals.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        written.append(str(path))
    return written


# --------------------------------------------------------------------------- stage-1 file -> tracks

def tracks_from_stage1(path, cfg: dict, with_meta: bool = False):
    """Build tracks from a stage-1 JSONL file (person_id, timestamp, bbox[, frame, conf, class]) so this
    module can run without the detector. Returns (tracks, det_classes), or (tracks, det_classes, meta) with
    with_meta (the meta line holds "stitch_pieces" for tracks joined by stitch.py)."""
    import features
    rows, det_classes = [], {}
    meta = {"fps": 30.0, "stride": 1, "proc_size": None}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "meta" in r:
                meta.update(r["meta"])
                continue
            frame = int(r.get("frame", round(float(r["timestamp"]) * float(meta["fps"]))))
            x1, y1, x2, y2 = r["bbox"]
            cls = int(r.get("class", 0))
            rows.append([frame, float(r["timestamp"]), int(r["person_id"]), x1, y1, x2, y2,
                         float(r.get("conf", 0.9)), cls])
            det_classes[(int(r["person_id"]), frame)] = cls
    res = {"fps": meta["fps"], "stride": meta["stride"], "proc_size": meta.get("proc_size"), "detections": rows}
    if with_meta:
        return features.build_tracks(res, cfg), det_classes, meta
    return features.build_tracks(res, cfg), det_classes


def write_jsonl(path, records) -> None:
    """One JSON object per line (NaN-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(utils._clean_nans(r), ensure_ascii=False) + "\n")


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: behaviour signals from a stage-1 tracking file")
    ap.add_argument("--tracks", required=True, help="stage1_tracking.jsonl")
    ap.add_argument("--zones", default=None)
    ap.add_argument("--keypoints", default=None, help="keypoints.json from a previous run (optional)")
    ap.add_argument("--scenario", default="pool")
    ap.add_argument("--out", default="stage2_behaviour.jsonl")
    ap.add_argument("--plots", action="store_true", help="also write one signal plot per swimmer next to --out")
    args = ap.parse_args(argv)
    cfg = utils.load_config(scenario=args.scenario)
    tracks, det_classes, meta = tracks_from_stage1(args.tracks, cfg, with_meta=True)
    proc = next(iter(tracks.values())).frame_size if tracks else None
    zones = utils.load_zones(args.zones, proc) if args.zones else []
    kps = utils.read_json(args.keypoints).get("keypoints") if args.keypoints else None
    import stitch                                             # joined tracks: each piece keeps its own body size
    samples = behaviour_samples(tracks, cfg, zones, kps, det_classes, pieces=stitch.piece_spans(meta))
    write_jsonl(args.out, samples)
    print(f"{len(samples)} behaviour samples for {len(tracks)} swimmers -> {args.out}")
    if args.plots and samples:
        for p in plot_signals(samples, Path(args.out).parent / "behaviour_plots"):
            print("  plot:", p)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
