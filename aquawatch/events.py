"""Final events: merge raw intervals, drop the too-short ones, score confidence, write the evidence.

NumPy only. This is where "a speed number" becomes "Person #3 ran from 00:41 to 00:48, because ...".
Each event carries its own numbers (metrics), a plain-English sentence (evidence) and a confidence
built from three explainable parts: margin past the threshold, duration, and detector confidence.

Events come in three shapes (DESIGN 4.10 lists the keys every event has):
  * one object ........ loitering, zone_intrusion, running, fall   (entity_id = that object)
  * a pair ............ near_miss   (entity_id = A, other_entity = B, entities = [A, B])
  * the scene ......... crowding    (entity_id None, entities = everyone counted)
"""
from __future__ import annotations

import approaching
import utils
from behaviors import compute_metrics, crowding_metrics
from utils import (BEHAVIORS, behavior_name, clip01, entity_name, entity_word, fmt_time, fmt_time_precise,
                   merge_intervals, unit_name)

_EPS = 1e-6   # so a span of exactly min_duration_s passes the ">= min_duration_s" rule


def min_duration_key(behavior: str) -> str:
    """Name of the minimum-duration setting of a behaviour (a fall calls it min_down_s)."""
    return "min_down_s" if behavior == "fall" else "min_duration_s"


def _min_duration(cfg: dict, behavior: str) -> float:
    """Minimum duration in seconds for a behaviour (0 when the config has none)."""
    return float(((cfg.get("behaviors") or {}).get(behavior) or {}).get(min_duration_key(behavior), 0.0))


def build_events(behavior_result: dict, tracks: dict, zones: list[dict], cfg: dict) -> dict:
    """Turn raw behaviour intervals into the final, report-ready events.

    Returns {"events": [...], "near_misses": [...], "unusual_tracks": [...], "baseline": {...}}.
    """
    ecfg = cfg["events"]
    merge_gap = float(ecfg["merge_gap_s"])
    ratio = float(ecfg["near_miss_ratio"])
    baseline = behavior_result.get("baseline") or {}
    baseline_tracks = baseline.get("tracks") or {}

    # 1. Group raw intervals. Pieces only merge when they belong together:
    #    one object + one behaviour (+ the zone it was in, or the other object of a near miss).
    groups: dict[tuple, dict] = {}
    for iv in behavior_result.get("intervals", []):
        key = _group_key(iv, tracks)
        if key is None:
            continue
        group = groups.setdefault(key, {"spans": [], "entities": set()})
        group["spans"].append((float(iv["start_s"]), float(iv["end_s"])))
        group["entities"].update(iv.get("entities") or [])

    events: list[dict] = []
    longest_dropped: dict[tuple, float] = {}                 # pairs whose pieces were all too short
    for key in sorted(groups, key=lambda k: (-1 if k[0] is None else k[0], k[1], str(k[2]))):
        tid, behavior, extra = key
        min_dur = _min_duration(cfg, behavior)
        for start, end in merge_intervals(groups[key]["spans"], merge_gap):    # 2. merge close pieces
            if (end - start) + _EPS < min_dur:                                  # 3. too short: drop
                pair = (tid, behavior)
                longest_dropped[pair] = max(longest_dropped.get(pair, 0.0), end - start)
                continue
            events.append(_make_event(tracks, tid, behavior, start, end, extra, groups[key]["entities"],
                                      zones, cfg, _lookup(baseline_tracks, tid)))

    # 4. Sort by start time (then person; scene events last) and number from 1.
    events.sort(key=lambda e: (e["start_s"], e["entity_id"] is None, e["entity_id"] or 0, e["behavior"]))
    for n, event in enumerate(events, start=1):
        event["event_id"] = n

    events, ignored = _apply_active_hours(events, cfg)

    pairs_with_events = {(e["entity_id"], e["behavior"]) for e in events}
    near_misses = _filter_near_misses(behavior_result.get("near_misses", []), pairs_with_events,
                                      longest_dropped, tracks, cfg, ratio)

    # "has_event": the object has an event of its own, or was the other party of a near miss.
    ids_with_events = {e["entity_id"] for e in events} | {e["other_entity"] for e in events}
    unusual = [{"entity_id": int(tid), "anomaly_score": info["anomaly_score"], "note": info["note"],
                "has_event": int(tid) in ids_with_events}
               for tid, info in baseline_tracks.items() if info.get("unusual")]
    unusual.sort(key=lambda u: (-u["anomaly_score"], u["entity_id"]))

    baseline_out = {k: v for k, v in baseline.items() if k != "tracks"}
    if not baseline_out:
        baseline_out = {"enabled": False, "active": False, "n_tracks": 0,
                        "median_speed_bh_s": 0.0, "median_dwell_s": 0.0}
    return {"events": events, "near_misses": near_misses, "unusual_tracks": unusual, "baseline": baseline_out,
            "ignored_outside_hours": ignored}


def _apply_active_hours(events: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    """Time-of-day rules: drop events outside their rule's `active_hours` (needs time_of_day.start_time).

    Every kept event gets its wall-clock times (clock_start / clock_end) when the start time is known.
    Returns (kept events renumbered from 1, short records of the ignored ones).
    """
    start_clock = utils.parse_clock(((cfg.get("time_of_day") or {}).get("start_time")) or "")
    if start_clock is None:
        return events, []
    kept, ignored = [], []
    for ev in events:
        ev["clock_start"] = utils.fmt_clock(start_clock + ev["start_s"])
        ev["clock_end"] = utils.fmt_clock(start_clock + ev["end_s"])
        spec = ((cfg["behaviors"].get(ev["behavior"]) or {}).get("active_hours")) or ""
        windows = utils.parse_hours(spec)
        if windows and not utils.span_in_hours(start_clock, ev["start_s"], ev["end_s"], windows):
            ignored.append({"behavior": ev["behavior"], "entity_id": ev["entity_id"],
                            "clock_start": ev["clock_start"], "clock_end": ev["clock_end"],
                            "active_hours": spec})
            continue
        if windows:
            ev["evidence"] += (f" Happened at {ev['clock_start']}, inside this rule's active hours ({spec}).")
        kept.append(ev)
    for n, ev in enumerate(kept, 1):
        ev["event_id"] = n
    return kept, ignored


def _group_key(iv: dict, tracks: dict):
    """(entity id, behaviour, zone / other id) that raw intervals are merged under; None to skip the interval."""
    behavior = iv.get("behavior")
    if behavior not in BEHAVIORS:
        return None
    tid = iv.get("track_id")
    if behavior == "crowding":
        return (None, behavior, iv.get("zone"))
    if tid is None or int(tid) not in tracks:
        return None
    tid = int(tid)
    if behavior == "zone_intrusion":
        return (tid, behavior, iv.get("zone"))
    if behavior == "near_miss":
        other = iv.get("other_id")
        if other is None or int(other) not in tracks:
            return None
        return (tid, behavior, int(other))
    return (tid, behavior, None)


# --------------------------------------------------------------------------- one event

def _make_event(tracks: dict, tid, behavior: str, start: float, end: float, extra, raw_entities,
                zones: list[dict], cfg: dict, baseline_info: dict | None) -> dict:
    """Metrics, confidence and evidence for one merged span.

    `extra` is the zone name (zone_intrusion, crowding) or the other object's id (near_miss).
    """
    zone = other_id = None
    if behavior == "crowding":
        metrics, members = crowding_metrics(tracks, zones, start, end, cfg, zone_name=extra)
        entities = members or sorted(int(e) for e in raw_entities)
        zone = metrics["zone"]
        name = f"Group of {len(entities)}"
        evidence_names = (None, None)
    elif behavior == "near_miss":
        a, b = tracks[tid], tracks[extra]
        metrics = compute_metrics(a, behavior, start, end, zones, cfg, other=b)
        entities, other_id = [int(tid), int(extra)], int(extra)
        name = entity_name(cfg, tid, a.cls)
        evidence_names = (name, entity_name(cfg, extra, b.cls))
    else:
        track = tracks[tid]
        metrics = compute_metrics(track, behavior, start, end, zones, cfg, zone_name=extra)
        entities = [int(tid)]
        zone = metrics.get("zone") if behavior == "zone_intrusion" else None
        name = entity_name(cfg, tid, track.cls)
        evidence_names = (name, None)

    parts = confidence_parts(behavior, metrics, cfg)
    weights = _weights(cfg)
    confidence = round(clip01(weights["margin"] * parts["margin"] + weights["duration"] * parts["duration"]
                              + weights["detection"] * parts["detection"]), 2)
    start_r, end_r = round(start, 2), round(end, 2)
    return {
        "event_id": 0,                                   # numbered after sorting
        "entity_id": None if tid is None else int(tid),
        "behavior": behavior,
        "start_s": start_r,
        "end_s": end_r,
        "start": fmt_time(start_r),
        "end": fmt_time(end_r),
        "duration_s": round(end - start, 2),
        "confidence": confidence,
        "confidence_parts": {k: round(v, 2) for k, v in parts.items()},
        "evidence": make_evidence(behavior, metrics, cfg, *evidence_names),
        "metrics": metrics,
        "zone": zone,
        "baseline": ({"anomaly_score": baseline_info["anomaly_score"], "note": baseline_info["note"]}
                     if baseline_info else None),
        "snapshot_time_s": metrics["peak_time_s"],
        "snapshot": None, "speed_plot": None,           # filled by render.py
        "verified": None, "pose": None,                 # filled by pose_verify.py
        "severity": None,                               # filled by chains.py
        "entities": entities,
        "other_entity": other_id,
        "entity_name": name,
        "behavior_name": behavior_name(cfg, behavior),
    }


def confidence_parts(behavior: str, metrics: dict, cfg: dict) -> dict[str, float]:
    """The three confidence scores, each in [0, 1].

    margin    how far past the rule the behaviour went (more extreme = more certain)
    duration  how long it lasted compared with twice the minimum duration
    detection mean YOLO detection confidence over the event
    """
    if behavior == "running":
        margin = clip01((metrics["mean_speed_bh_s"] / metrics["start_threshold_bh_s"] - 1.0) / 0.5)
    elif behavior == "loitering":
        # DESIGN says max_radius_bh; we use the 90th-percentile radius when available (see
        # behaviors.compute_metrics) because walk-in / walk-out steps inflate the plain maximum.
        radius = metrics.get("p90_radius_bh", metrics["max_radius_bh"])
        margin = clip01(1.0 - radius / metrics["radius_threshold_bh"])
    elif behavior == "fall":
        down_min = float(cfg["behaviors"]["fall"]["down_min_aspect"])
        margin = clip01((metrics["max_aspect"] / down_min - 1.0) / 0.5)
        if metrics["fell"]:
            margin = min(1.0, margin + 0.2)               # a visible drop is stronger evidence than lying
    elif behavior == "crowding":
        need = float(cfg["behaviors"]["crowding"]["min_count"])
        margin = clip01((metrics["max_count"] / need - 1.0) / 0.5)
    elif behavior == "near_miss":
        near = float(cfg["behaviors"]["near_miss"]["near_distance_bh"])
        margin = clip01(1.0 - metrics["min_distance_bh"] / near) if metrics["min_distance_bh"] is not None else 0.0
        if metrics["min_ttc_s"] is not None:
            margin += 0.3 * clip01((1.5 - metrics["min_ttc_s"]) / 1.5)
        margin = clip01(margin)
    elif behavior == "approaching":
        margin = approaching.approaching_margin(metrics, cfg)
    else:  # zone_intrusion
        margin = clip01(metrics["max_depth_bh"] / 0.5)
    min_dur = metrics.get(min_duration_key(behavior), 0.0)
    duration = clip01(metrics["duration_s"] / (2.0 * min_dur)) if min_dur > 0 else 1.0
    return {"margin": margin, "duration": duration, "detection": clip01(metrics["mean_det_conf"])}


def _weights(cfg: dict) -> dict[str, float]:
    """Confidence weights from config, scaled to sum to 1 (a no-op for the default 0.4/0.3/0.3)."""
    w = cfg["events"]["confidence_weights"]
    parts = {k: max(0.0, float(w.get(k, 0.0))) for k in ("margin", "duration", "detection")}
    total = sum(parts.values())
    return {k: v / total for k, v in parts.items()} if total > 0 else {k: 1.0 / 3.0 for k in parts}


def _short_unit(cfg: dict | None) -> str:
    """"bh" for body-heights, "bl" for body-lengths."""
    return "".join(word[0] for word in unit_name(cfg).split("-") if word) or "u"


def _plural(word: str) -> str:
    """Plural of the scenario's entity word, in lower case: "Person" -> "people", "Animal" -> "animals"."""
    word = word.strip().lower()
    if word == "person":
        return "people"
    return word if word.endswith("s") else word + "s"


def make_evidence(behavior: str, m: dict, cfg: dict | None = None, entity: str | None = None,
                  other: str | None = None) -> str:
    """One plain-English sentence with the measured numbers and the rule that was applied.

    `cfg` gives the scenario wording (unit and entity word) and the rule thresholds; `entity` / `other`
    are the display names of the objects of a near miss (e.g. "Person #3", "Car #7").
    """
    cfg = cfg or {}
    unit, short = unit_name(cfg), _short_unit(cfg)
    if behavior == "loitering":
        return (f"Stayed within {m['max_radius_bh']:.2f} {unit} of one spot for {m['duration_s']:.0f} s. "
                f"Rule: within {m['radius_threshold_bh']:g} {short} for >= {m['min_duration_s']:g} s.")
    if behavior == "zone_intrusion":
        return (f"Feet inside zone '{m['zone']}' for {m['duration_s']:.1f} s, "
                f"up to {m['max_depth_bh']:.2f} {unit} deep. Rule: >= {m['min_duration_s']:g} s.")
    if behavior == "fall":
        if m["fell"]:
            return (f"Box went from upright (w/h {m['upright_aspect_before']:.1f}) to lying "
                    f"(w/h {m['max_aspect']:.1f}) within {m['transition_s']:.1f} s and stayed down "
                    f"{m['duration_s']:.0f} s.")
        return f"Lying (w/h {m['max_aspect']:.1f}) for {m['duration_s']:.0f} s."
    if behavior == "crowding":
        rule = (cfg.get("behaviors") or {}).get("crowding") or {}
        return (f"{m['max_count']} {_plural(entity_word(cfg))} (avg {m['mean_count']:.1f}) inside "
                f"'{m['zone']}' for {m['duration_s']:.0f} s. "
                f"Rule: >= {rule.get('min_count', '?')} for >= {m['min_duration_s']:g} s.")
    if behavior == "near_miss":
        return _near_miss_evidence(m, cfg, unit, entity, other)
    if behavior == "approaching":
        return f"{entity or 'Object'} " + approaching.approaching_evidence(m, cfg)
    return (f"Moved at {m['mean_speed_bh_s']:.1f} {unit}/s (peak {m['peak_speed_bh_s']:.1f}) "
            f"for {m['duration_s']:.1f} s. Rule: >= {m['start_threshold_bh_s']:g} {short}/s for "
            f">= {m['min_duration_s']:g} s; walking is about 0.6-1.0.")


def _near_miss_evidence(m: dict, cfg: dict, unit: str, entity: str | None, other: str | None) -> str:
    """Sentence for a near miss: who, how close, how fast, whether they touched."""
    rule = (cfg.get("behaviors") or {}).get("near_miss") or {}
    a = entity or entity_name(cfg, "A")
    b = other or entity_name(cfg, m["other_id"], m.get("other_class"))
    if m.get("min_distance_bh") is None:
        return f"{a} and {b} came very close to each other."
    when = fmt_time_precise(m["peak_time_s"])
    text = (f"{a} and {b} came within {m['min_distance_bh']:.2f} {unit} of each other at {when}, "
            f"closing at {m['max_closing_speed_bh_s']:.1f} {unit}/s")
    if m["contact"]:
        text += " (possible contact)"
    elif m["min_ttc_s"] is not None:
        text += f" (time-to-collision {m['min_ttc_s']:.1f} s)"
    text += f". They approached for {m['approach_s']:.1f} s, then separated."
    if rule:
        text += (f" Rule: closer than {rule.get('near_distance_bh', '?')} {unit} while moving >= "
                 f"{rule.get('min_rel_speed_bh_s', '?')} {unit}/s relative.")
    return text


def _lookup(baseline_tracks: dict, tid):
    """Baseline entry of a track; keys may be ints (in memory) or strings (after a JSON round trip)."""
    if tid is None:
        return None
    return baseline_tracks.get(tid) or baseline_tracks.get(str(tid))


# --------------------------------------------------------------------------- near misses

def _filter_near_misses(raw_near: list[dict], pairs_with_events: set, longest_dropped: dict,
                        tracks: dict, cfg: dict, ratio: float) -> list[dict]:
    """Keep "almost flagged" cases for objects that were NOT flagged for that behaviour.

    * a near miss for a (object, behaviour) pair that did produce an event is dropped (the event
      already says it all);
    * a pair whose pieces were all shorter than the minimum duration is re-described by its merged
      length, and kept if it reached `ratio` of the minimum ("too short, not flagged"). Only the
      single-object behaviours (loitering, zone_intrusion, running, fall) are described this way.
    """
    out = [dict(nm) for nm in raw_near
           if (nm["track_id"], nm["behavior"]) not in pairs_with_events
           and (nm["track_id"], nm["behavior"]) not in longest_dropped]
    for (tid, behavior), length in longest_dropped.items():
        if tid is None or behavior in ("crowding", "near_miss"):
            continue                                     # scene / pair events have no "person" to blame
        if (tid, behavior) in pairs_with_events:
            continue                                     # another span of the pair became an event
        need = _min_duration(cfg, behavior)
        if length + _EPS < ratio * need:
            continue                                     # too short to be an "almost"
        out.append({"track_id": tid, "behavior": behavior, "value": round(length, 1), "threshold": need,
                    "unit": "s", "note": _too_short_note(tid, behavior, length, need, tracks, cfg)})
    out.sort(key=lambda nm: (nm["track_id"], nm["behavior"]))
    return out


def _too_short_note(tid: int, behavior: str, length: float, need: float, tracks: dict, cfg: dict) -> str:
    """Sentence for a near miss whose only problem was the duration."""
    who = entity_name(cfg, tid, tracks[tid].cls if tid in tracks else None)
    if behavior == "running":
        speed = float(cfg["behaviors"]["running"]["start_speed_bh_s"])
        return (f"{who} ran for {length:.1f} s at >= {speed:g} {unit_name(cfg)}/s "
                f"(rule: >= {need:g} s). Too short, not flagged.")
    if behavior == "loitering":
        return f"{who} stood still for {length:.1f} s (rule: {need:g} s). Too short, not flagged."
    if behavior == "fall":
        return f"{who} was lying down for {length:.1f} s (rule: >= {need:g} s). Too short, not flagged."
    return (f"{who} was inside a restricted zone for {length:.1f} s "
            f"(rule: >= {need:g} s). Too short, not flagged.")
