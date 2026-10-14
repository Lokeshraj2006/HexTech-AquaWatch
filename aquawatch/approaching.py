"""Approaching-object alert for a MOVING (wearable / chest-mounted) camera.

Why a separate rule: on a moving camera every object slides across the image when the wearer
walks, so position-based speed, zones and loitering stop meaning anything. What still works is
LOOMING: an object coming at you gets bigger in the image. If its size is s, then

    growth rate  g = d(ln s)/dt          (per second)
    time to contact  TTC = 1 / g         (seconds, when g > 0)

because the image size of an object is inversely proportional to its distance. Size change is
independent of the camera's sideways motion, so this works while the wearer walks.

An alert fires when an object is (1) growing fast enough that TTC <= `approaching.ttc_s`,
(2) in the wearer's path AT THE MOMENT OF CONTACT (people in the next lane, who drift outward as
you get closer, are not flagged), (3) growing uniformly (width and height together, so raising
arms is not an approach), and (4) already
close enough to matter (its size is a sizeable fraction of the frame height), for at least
`approaching.min_duration_s`.

Pure NumPy (no OpenCV / torch), like the other reasoning modules. See docs/DESIGN.md section 4.11.
"""
from __future__ import annotations

import math

import numpy as np

import utils


# --------------------------------------------------------------------------- signals

def looming_signals(track, cfg: dict, frame_size) -> dict:
    """Per-sample looming signals for one track.

    Returns arrays (same length as the track): growth (1/s, NaN where unknown), ttc (s, inf when
    not approaching), center_offset (-1 = left edge, 0 = centre, +1 = right edge), predicted
    offset at contact, size_frac (object size / frame height) and the boolean `alert` mask.
    """
    acfg = cfg["behaviors"]["approaching"]
    t = np.asarray(track.t, dtype=float)
    n = len(t)
    width, height = float(frame_size[0]), float(frame_size[1])
    box = np.asarray(track.box, dtype=float).reshape(-1, 4)
    scale = np.asarray(track.height, dtype=float)               # smoothed body scale (px)
    window = float(acfg.get("window_s", 0.5))

    # Growth of ln(size) over a trailing window, within one track segment (no growth across gaps).
    # Width and height are also tracked separately: a real approach makes BOTH grow at about the
    # same rate (the whole object gets bigger), while raising arms or bending changes only one.
    bw = np.maximum(box[:, 2] - box[:, 0], 1.0)
    bh = np.maximum(box[:, 3] - box[:, 1], 1.0)
    growth = np.full(n, np.nan)
    growth_w = np.full(n, np.nan)
    growth_h = np.full(n, np.nan)
    segment = np.asarray(track.segment)
    lo = np.searchsorted(t, t - window, side="left")
    for i in range(n):
        j = lo[i]
        while j < i and segment[j] != segment[i]:
            j += 1
        dt = t[i] - t[j]
        if dt >= 0.5 * window and scale[i] > 0 and scale[j] > 0:
            growth[i] = (math.log(scale[i]) - math.log(scale[j])) / dt
            growth_w[i] = (math.log(bw[i]) - math.log(bw[j])) / dt
            growth_h[i] = (math.log(bh[i]) - math.log(bh[j])) / dt

    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(growth > float(acfg.get("min_growth_per_s", 0.05)), 1.0 / growth, np.inf)
        ratio = growth_w / growth_h
    tol = float(acfg.get("uniform_growth_ratio", 2.0))
    uniform = (growth_w > 0) & (growth_h > 0) & (ratio <= tol) & (ratio >= 1.0 / tol)

    # Where is it horizontally, and where will it be at the moment of contact? Only the position
    # AT CONTACT decides "in my path": someone in the next lane looks central when far away but
    # drifts outward as you get closer, and will pass beside you.
    cx = (box[:, 0] + box[:, 2]) / 2.0
    half = width / 2.0
    center_offset = (cx - half) / half
    vel = getattr(track, "vel", None)
    vx = np.asarray(vel, dtype=float)[:, 0] if vel is not None else np.full(n, np.nan)
    finite_ttc = np.where(np.isfinite(ttc), np.minimum(ttc, 10.0), 0.0)
    predicted = np.where(np.isfinite(vx), center_offset + vx * finite_ttc / half, center_offset)

    edge = getattr(track, "at_edge", None)
    at_edge = np.asarray(edge, dtype=bool) if edge is not None else np.zeros(n, dtype=bool)
    size_frac = scale / height
    in_path = np.abs(predicted) <= float(acfg["center_band"])
    alert = ((ttc <= float(acfg["ttc_s"])) & in_path & uniform & ~at_edge
             & (size_frac >= float(acfg["min_size_frac"])))
    return {"t": t, "growth": growth, "growth_w": growth_w, "growth_h": growth_h, "uniform": uniform,
            "ttc": ttc, "center_offset": center_offset,
            "predicted_offset": predicted, "size_frac": size_frac, "alert": alert}


def clock_direction(center_offset: float) -> str:
    """Clock position of an object in front of the wearer: 10 (far left) .. 12 (ahead) .. 2 (far right)."""
    if not math.isfinite(center_offset):
        return "12 o'clock"
    # Map -1..+1 across the image to roughly 10 o'clock .. 2 o'clock (a ~120 degree field of view).
    hour = 12 + round(center_offset * 2)
    hour = 12 if hour == 12 else (hour if hour < 12 else hour - 12)
    return f"{hour} o'clock"


# --------------------------------------------------------------------------- rule

def applies(track, cfg: dict) -> bool:
    """Is the rule on, and is this object's class one that should trigger an alert?"""
    acfg = cfg["behaviors"].get("approaching") or {}
    return bool(acfg.get("enabled")) and int(getattr(track, "cls", 0)) in [int(c) for c in acfg.get("classes", [0])]


def approaching_intervals(track, cfg: dict, frame_size=None) -> list[tuple[float, float]]:
    """Raw (start_s, end_s) runs where the object is on course to reach the wearer soon.

    Duration filtering happens later in events.py (after merging), like the other rules.
    """
    frame_size = frame_size or getattr(track, "frame_size", None)
    if not frame_size or not applies(track, cfg) or len(track.t) < 3:
        return []
    sig = looming_signals(track, cfg, frame_size)
    runs = utils.mask_to_intervals(sig["t"], sig["alert"], max_gap_s=cfg["features"]["max_gap_s"])
    spans = []
    for start_s, end_s, _, _ in runs:
        if end_s - start_s < 1e-6:                      # a single sample: give it a small span
            start_s, end_s = start_s - 0.15, end_s + 0.15
        spans.append((max(0.0, start_s), end_s))
    # One alert per object per cooldown: repeated warnings about the same person are just noise.
    cooldown = float(cfg["behaviors"]["approaching"].get("cooldown_s", 3.0))
    return utils.merge_intervals(spans, cooldown)


def approaching_metrics(track, start_s: float, end_s: float, cfg: dict, frame_size=None) -> dict:
    """Numbers for one approaching event (the merged span): closest time to contact, where, how big."""
    acfg = cfg["behaviors"]["approaching"]
    frame_size = frame_size or getattr(track, "frame_size", None)
    sig = looming_signals(track, cfg, frame_size)
    idx = np.where((sig["t"] >= start_s - 1e-6) & (sig["t"] <= end_s + 1e-6))[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(sig["t"] - (start_s + end_s) / 2)))])
    ttc = sig["ttc"][idx]
    k = idx[int(np.argmin(ttc))] if np.isfinite(ttc).any() else idx[len(idx) // 2]
    min_ttc = float(sig["ttc"][k]) if np.isfinite(sig["ttc"][k]) else None
    offset = float(sig["center_offset"][k])
    who = utils.entity_name(cfg, track.track_id, getattr(track, "cls", 0)).split(" #")[0]
    direction = clock_direction(offset)
    alert_text = (f"{who} approaching from {direction}"
                  + (f", about {min_ttc:.1f} seconds away." if min_ttc is not None else "."))
    return {
        "duration_s": round(float(end_s - start_s), 2),
        "min_ttc_s": None if min_ttc is None else round(min_ttc, 2),
        "max_growth_per_s": round(float(np.nanmax(sig["growth"][idx])), 3) if np.isfinite(sig["growth"][idx]).any() else None,
        "size_frac_at_alert": round(float(sig["size_frac"][k]), 3),
        "center_offset": round(offset, 3),
        "direction": direction,
        "alert_text": alert_text,
        "ttc_threshold_s": float(acfg["ttc_s"]),
        "min_duration_s": float(acfg["min_duration_s"]),
        "peak_time_s": float(sig["t"][k]),
        "mean_det_conf": round(float(np.nanmean(np.asarray(track.conf, dtype=float)[idx])), 3),
    }


def approaching_evidence(metrics: dict, cfg: dict) -> str:
    """Plain-English evidence sentence for an approaching event."""
    growth = metrics.get("max_growth_per_s")
    grow_txt = f"grew {growth * 100:.0f}%/s in view" if growth is not None else "grew quickly in view"
    ttc = metrics.get("min_ttc_s")
    ttc_txt = f"time to contact about {ttc:.1f} s" if ttc is not None else "time to contact unknown"
    return (f"{grow_txt} at {metrics['direction']} ({ttc_txt}) at "
            f"{utils.fmt_time_precise(metrics['peak_time_s'])}. Rule: time to contact <= "
            f"{metrics['ttc_threshold_s']:.1f} s while in the walking path.")


def approaching_margin(metrics: dict, cfg: dict) -> float:
    """Confidence margin in [0, 1]: how far below the time-to-contact threshold it got."""
    ttc = metrics.get("min_ttc_s")
    limit = float(cfg["behaviors"]["approaching"]["ttc_s"])
    return 0.0 if ttc is None else utils.clip01((limit - ttc) / limit + 0.3)


def approaching_severity(metrics: dict) -> str:
    """high when contact is about a second away, else medium."""
    ttc = metrics.get("min_ttc_s")
    return "high" if ttc is not None and ttc <= 1.0 else "medium"
