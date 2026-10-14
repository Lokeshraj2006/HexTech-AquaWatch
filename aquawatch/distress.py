"""Stage 3 of the pool pipeline: is this behaviour actually BECOMING dangerous? (temporal reasoning)

Input: the stage-2 behaviour samples of every swimmer (pool_behaviour.py). Output: events in the
team's stage-3 data contract, plus a risk curve and a state timeline per swimmer for the dashboard:

    {"person_id": 3, "event": "high_risk_aquatic_distress", "start_time": 79.2, "alert_time": 84.2,
     "risk_score": 0.92, "location": "Deep End",
     "evidence": ["vertical_posture", "low_displacement", "repeated_arm_motion", "transition_from_swimming"]}

How it decides (no single frame and no single signal decides anything):

1. Every sample gets a RISK = sum of the warning signs present (weights in config pool.weights, max 1.0):
   vertical posture, no forward progress, repeated arm motion, unstable / submerged head, and
   "transition" (the same swimmer was swimming normally within the last pool.lookback_s seconds).
2. Look-alikes are damped: at the wall with calm arms = resting (risk x pool.resting_factor);
   upright with calm arms and a steady head = treading water (risk capped at pool.treading_cap);
   anyone outside the water zone is ignored.
3. A per-swimmer STATE MACHINE turns the risk curve into states:
   NORMAL -> WATCH (risk >= watch_risk) -> WARNING (>= warning_risk) -> DISTRESS when risk stays
   >= alert_risk for hold_s seconds (it only ends when risk falls below release_risk: hysteresis).
   The alert is raised hold_s after the distress run began.
4. SUBMERSION: a swimmer in WARNING or DISTRESS who disappears for submersion_lost_s seconds (or whose
   head is "submerged" for submersion_head_s) -> "possible_submersion". A dive from normal swimming
   does not count, because no warning came first.

NumPy-free pure Python. Run on its own: python distress.py --behaviour stage2_behaviour.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import utils

STATES = ("NORMAL", "WATCH", "WARNING", "DISTRESS", "SUBMERSION")
SIGNAL_NOTES = {
    "slowing": "Movement slowing",
    "vertical_posture": "Vertical posture",
    "repeated_arm_motion": "Repeated arm movement",
    "low_displacement": "Very low displacement",
    "unstable_head": "Head position unstable",
    "submerged_head": "Head going under",
    "stationary_upright": "Upright and not moving for a long time",
    "passive_still": "Still and face-down for a long time",
}


def sample_signals(s: dict, pcfg: dict) -> dict:
    """Which warning signs are present in one behaviour sample, and is it one of the normal look-alikes."""
    low_disp = float(s["displacement"]) <= float(pcfg["displacement_low_bh_s"])
    # "diagonal" (tilted torso) counts as upright only when the swimmer is not getting anywhere
    vertical = s["posture"] in ("vertical", "head_only", "underwater") or (s["posture"] == "diagonal" and low_disp)
    # Arm strokes are normal while swimming; the distress "climbing the ladder" motion happens UPRIGHT.
    arms = s["arm_motion"] == "repeated" and vertical
    head_bad = s["head"] in ("unstable", "submerged")
    swimming = (s["posture"] == "horizontal" and s["movement"] in ("medium", "high") and not low_disp)
    resting = bool(s.get("near_wall")) and not arms
    # treading / standing = upright with NO active sign (config.yaml pool.treading_cap); 'irregular' arms or an
    # 'unknown' head are not signs (bug hunt 2026-10-11, pool high; PREREGISTRATION_bug_group_B.md step B1)
    treading = vertical and not arms and not head_bad
    return {"vertical_posture": vertical, "low_displacement": low_disp, "repeated_arm_motion": arms,
            "unstable_head": head_bad, "submerged_head": s["head"] == "submerged",
            "swimming_normally": swimming, "resting_at_wall": resting, "treading_water": treading,
            "slowing": s["posture"] == "horizontal" and s["movement"] == "low",
            # the trained detector's out_of_water state = on the deck: not judged, like outside the water zone
            "in_water": bool(s.get("in_water", True)) and s["posture"] != "out_of_water"}


def sample_risk(sig: dict, transition: bool, pcfg: dict, treading_cap: bool = True) -> float:
    """Risk of one sample in [0, 1]: weighted warning signs, then the look-alike damping. treading_cap=False leaves
    out the treading-water cap (used only to decide whether an alarm already raised ends, see _reason_one)."""
    if not sig["in_water"]:
        return 0.0
    w = pcfg["weights"]
    r = (w["vertical"] * sig["vertical_posture"] + w["low_displacement"] * sig["low_displacement"]
         + w["repeated_arms"] * sig["repeated_arm_motion"] + w["unstable_head"] * sig["unstable_head"]
         + w["transition"] * (transition and sig["vertical_posture"])
         + float(w.get("stationary", 0.0)) * sig.get("stationary_upright", False))
    if sig.get("passive_still"):                 # still and face-down for passive_s: at least an early warning
        r = max(r, float(pcfg["warning_risk"]))
    if sig["resting_at_wall"]:
        r *= float(pcfg["resting_factor"])
    # Upright + calm arms + steady head = standing or treading water: risk capped. Upright and still is not enough
    # on its own; it needs an active sign. (stationary_lifts_treading_cap = the old rule, see config.yaml.)
    lifted = bool(pcfg.get("stationary_lifts_treading_cap", False)) and sig.get("stationary_upright")
    if treading_cap and sig["treading_water"] and not lifted:
        r = min(r, float(pcfg["treading_cap"]))
    return round(min(1.0, r), 3)


def reason(samples: list[dict], cfg: dict, last_seen: dict | None = None, video_end: float | None = None,
           gaps: dict | None = None, vanished: dict | None = None, same_spot: dict | None = None) -> dict:
    """Run the state machine for every swimmer.

    `last_seen`: {person_id: last time the detector saw them}; `video_end`: video length in seconds
    (both used for "disappeared while in distress" = possible submersion). `gaps`: {person_id: [(last seen,
    seen again), ...]} for gaps of at least pool.submersion_lost_s INSIDE a track (a track joined by stitch.py
    can contain the moment the swimmer went under): vanishing there while in DISTRESS is a possible submersion too.
    `vanished`: {person_id: {"lost_at": s, ...}} from pool.vanished_in_water (Rule A): an early warning
    "lost from view in the water" for a swimmer who was not already in trouble. `same_spot`: {person_id: {"lost_at",
    "back_at", ...}} from pool.back_at_same_spot (Rule C): an early warning "seen again at the same spot".
    Returns {"events": [...], "risk": {pid: [[t, risk], ...]}, "timeline": {pid: [...]}, "status": {pid: ...}}.
    """
    pcfg = cfg["pool"]
    warn_on = bool(((cfg.get("behaviors") or {}).get("aquatic_warning") or {}).get("enabled", False))
    by_person: dict[int, list[dict]] = {}
    for s in samples:
        by_person.setdefault(int(s["person_id"]), []).append(s)
    events, risk_curves, timelines, status = [], {}, {}, {}
    # also swimmers with a Rule A / C finding but no behaviour sample (seen too briefly to be sampled): the finding
    # must not be dropped (bug hunt 2026-10-11, pool medium; PREREGISTRATION_bug_group_B.md step B5)
    for pid in sorted(set(by_person) | set(vanished or {}) | set(same_spot or {})):
        seq = sorted(by_person.get(pid, []), key=lambda s: s["timestamp"])
        res = _reason_one(pid, seq, pcfg, (last_seen or {}).get(pid), video_end, (gaps or {}).get(pid) or [],
                          warn_on, (vanished or {}).get(pid), (same_spot or {}).get(pid))
        events += res["events"]
        risk_curves[pid] = res["risk"]
        timelines[pid] = res["timeline"]
        status[pid] = res["final_state"]
    events.sort(key=lambda e: (e["start_time"], e["person_id"]))
    return {"events": events, "risk": risk_curves, "timeline": timelines, "status": status}


def _reason_one(pid: int, seq: list[dict], pcfg: dict, last_seen, video_end, gaps=(), warn_on: bool = False,
                vanished: dict | None = None, same_spot: dict | None = None) -> dict:
    """State machine for one swimmer over its time-ordered samples (`gaps`: see reason(); `warn_on`: also raise
    "early_warning" events, behaviour aquatic_warning; `vanished`: this swimmer's Rule A finding, or None)."""
    gaps = sorted(gaps)
    hold, look = float(pcfg["hold_s"]), float(pcfg["lookback_s"])
    warn_risk = float(pcfg["warning_risk"])
    ew_window, ew_fraction = float(pcfg["early_warning_window_s"]), float(pcfg["early_warning_fraction"])
    ew_cooldown = float(pcfg["early_warning_cooldown_s"])
    recent: list[tuple] = []            # (t, risk, evidence) of the valid samples of the last early_warning_window_s
    last_warning = None
    stationary_s, passive_s = float(pcfg["stationary_s"]), float(pcfg["passive_s"])
    standing_words = [str(w).lower() for w in (pcfg.get("standing_zone_words") or [])]
    prev_t = still_since = passive_since = None
    last_normal = None                  # last time this swimmer was swimming normally
    state, run_start, run_risks, run_evidence = "NORMAL", None, [], set()
    head_sub_since = None
    events, risk_curve, timeline, seen_notes = [], [], [], set()
    distress_event = None

    def lost_inside(lost_at: float, seen_again: float, where: str) -> bool:
        """Possible submersion where a (joined) track was lost while in DISTRESS, exactly as if that piece had ended
        there. Returns True when the swimmer stayed out of view for submersion_lost_s (the distress episode is
        over); a shorter gap (an ID switch) keeps them in DISTRESS, so a later real disappearance alerts again."""
        events.append(_submersion(pid, lost_at, where, "lost_from_view"))
        timeline.append({"t": round(lost_at, 2), "state": "SUBMERSION", "note": "Possible submersion (lost from view)"})
        return seen_again - lost_at >= float(pcfg["submersion_lost_s"])

    for s in seq:
        t = float(s["timestamp"])
        # vanished INSIDE the track (one joined by stitch.py) while in DISTRESS = possible submersion, like at its end.
        # (sample times are rounded to 0.01 s, so compare at that precision: never judge a gap before its own sample)
        while gaps and round(gaps[0][0], 2) < t:
            lost_at, seen_again = gaps.pop(0)
            if state == "DISTRESS":
                where = distress_event["location"] if distress_event else s["location"]
                if lost_inside(lost_at, seen_again, where):
                    state, run_start, run_risks, run_evidence, distress_event = "SUBMERSION", None, [], set(), None
        if s.get("track_valid") is False:          # ID switch suspected: this window mixes two people
            continue
        sig = sample_signals(s, pcfg)
        # How long has this swimmer been upright and not moving, or flat, still and face-down? A gap in the samples
        # (not seen) starts the count again. Never at the wall or in a zone where standing still is normal.
        if prev_t is not None and t - prev_t > 2 * float(pcfg["sample_s"]) + 0.05:
            still_since = passive_since = None
            # hold_s must be HELD while seen: a gap with no valid sample (not seen, or an ID switch) also restarts a
            # distress run that has not become an alarm yet (bug hunt 2026-10-11, pool; bug group B step B9)
            if state != "DISTRESS":
                run_start, run_risks, run_evidence = None, [], set()
        prev_t = t
        quiet = (sig["in_water"] and not s.get("near_wall") and sig["low_displacement"]
                 and not any(word in str(s.get("location") or "").lower() for word in standing_words))
        still_since = (t if still_since is None else still_since) if quiet and sig["vertical_posture"] else None
        # face NOT seen (unknown) or under: a head that was seen bobbing is a floater, not passive drowning
        # (bug hunt 2026-10-11, pool medium; step B4)
        flat_still = quiet and s["posture"] == "horizontal" and s["head"] in ("unknown", "submerged")
        passive_since = (t if passive_since is None else passive_since) if flat_still else None
        sig["stationary_upright"] = still_since is not None and t - still_since >= stationary_s - 1e-9
        sig["passive_still"] = passive_since is not None and t - passive_since >= passive_s - 1e-9
        if sig["swimming_normally"]:
            last_normal = t
        transition = last_normal is not None and (t - last_normal) <= look
        r = sample_risk(sig, transition, pcfg)
        risk_curve.append([round(t, 2), r])

        # timeline notes: the first time each warning sign shows up after normal swimming
        if sig["swimming_normally"] and "swimming" not in seen_notes:
            timeline.append({"t": round(t, 2), "state": "NORMAL", "note": "Normal swimming"})
            seen_notes.add("swimming")
        # (warning signs are only logged once the risk is rising, so a lap swimmer's turn is not "news")
        if sig["in_water"] and not sig["resting_at_wall"] and not sig["treading_water"] and                 (r >= float(pcfg["watch_risk"]) or sig["slowing"]):
            for key, note in SIGNAL_NOTES.items():
                if sig.get(key) and key not in seen_notes and (key != "slowing" or "swimming" in seen_notes):
                    timeline.append({"t": round(t, 2), "state": _state_for(r, pcfg), "note": note})
                    seen_notes.add(key)

        # head under water for long enough while already in trouble = submersion
        if sig["submerged_head"]:
            head_sub_since = t if head_sub_since is None else head_sub_since
        else:
            head_sub_since = None

        if state == "DISTRESS":
            # an alarm already raised is not ended by the treading cap: a drowning person's last seconds can look
            # still and calm (PREREGISTRATION_bug_group_B.md amendment B1b)
            if max(r, sample_risk(sig, transition, pcfg, treading_cap=False)) >= float(pcfg["release_risk"]):
                run_risks.append(r)
                run_evidence |= _evidence(sig, transition)
                distress_event["end_time"] = round(t, 2)
                # never lower than at the alarm: weaker samples later (kept in DISTRESS by hysteresis) must not pull
                # it under announce.min_risk and silence the spoken alarm (bug hunt, pool medium; step B3)
                distress_event["risk_score"] = max(distress_event["risk_score"],
                                                   _risk_score(run_risks, t - run_start, hold))
                distress_event["evidence"] = _ordered(run_evidence)
            else:
                state, run_start, run_risks, run_evidence = _state_for(r, pcfg), None, [], set()
                distress_event = None
        else:
            if r >= float(pcfg["alert_risk"]):
                if run_start is None:
                    run_start, run_risks, run_evidence = t, [], set()
                run_risks.append(r)
                run_evidence |= _evidence(sig, transition)
                if t - run_start >= hold - 1e-9:
                    state = "DISTRESS"
                    distress_event = {
                        "person_id": pid, "event": "high_risk_aquatic_distress",
                        "start_time": round(run_start, 2), "alert_time": round(t, 2), "end_time": round(t, 2),
                        "risk_score": _risk_score(run_risks, t - run_start, hold),
                        "evidence": _ordered(run_evidence), "location": s["location"]}
                    events.append(distress_event)
                    timeline.append({"t": round(t, 2), "state": "DISTRESS", "note": "HIGH-RISK AQUATIC DISTRESS"})
            else:
                run_start, run_risks, run_evidence = None, [], set()
                state = _state_for(r, pcfg)

        # EARLY WARNING ("please check swimmer N, they may need help"): the risk is at WARNING level in at least
        # early_warning_fraction of the samples of the last early_warning_window_s, before (never during) a DISTRESS
        # alarm. A struggle whose signs come and go (a hand-held camera, splashing that hides the arms) often stays
        # just under the 0.70 distress level: the lifeguard is still asked to look. Once per swimmer per cooldown.
        recent = [x for x in recent if x[0] > t - ew_window] + [(t, r, _evidence(sig, transition))]
        hot = [x for x in recent if x[1] >= warn_risk]
        if (warn_on and state not in ("DISTRESS", "SUBMERSION") and r >= warn_risk and len(recent) >= 4
                and len(hot) >= ew_fraction * len(recent)
                and (last_warning is None or t - last_warning >= ew_cooldown)):
            events.append({"person_id": pid, "event": "early_warning", "start_time": round(hot[0][0], 2),
                           "alert_time": round(t, 2), "end_time": round(t, 2),
                           "risk_score": round(sum(x[1] for x in hot) / len(hot), 2),
                           "evidence": _ordered(set().union(*(x[2] for x in hot))), "location": s["location"]})
            timeline.append({"t": round(t, 2), "state": "WARNING", "note": "EARLY WARNING: may need help"})
            last_warning = t

        if (head_sub_since is not None and state == "DISTRESS"
                and t - head_sub_since >= float(pcfg["submersion_head_s"])):
            events.append(_submersion(pid, head_sub_since, s["location"], "head_submerged"))
            timeline.append({"t": round(head_sub_since, 2), "state": "SUBMERSION", "note": "Possible submersion"})
            state, head_sub_since = "SUBMERSION", None

    # Joins AFTER the last sample (the next piece was too short or flickering to give a sample of its own): judged now.
    for lost_at, seen_again in gaps:
        if state == "DISTRESS":
            where = distress_event["location"] if distress_event else seq[-1]["location"]
            if lost_inside(lost_at, seen_again, where):
                state = "SUBMERSION"

    # Vanished while in trouble (the detector lost them in the water) = possible submersion.
    last_t = float(seq[-1]["timestamp"]) if seq else 0.0
    gone_at = float(last_seen) if last_seen is not None else last_t
    still_video = video_end is None or (float(video_end) - gone_at) >= float(pcfg["submersion_lost_s"])
    # Only a swimmer already in DISTRESS who then vanishes counts: people who were merely "warning" (playing,
    # treading) disappear all the time by swimming out of view or behind others (seen on real pool footage).
    # (Being in DISTRESS already means no submersion was raised since this distress episode began: every submersion
    # sets the state to SUBMERSION. An EARLIER submersion, e.g. the swimmer went under, came back up and struggled
    # again, must not silence this one: the final disappearance is the most dangerous moment.)
    in_trouble = state == "DISTRESS"
    if state != "SUBMERSION" and seq and in_trouble and still_video and seq[-1].get("in_water", True):
        events.append(_submersion(pid, gone_at, seq[-1]["location"], "lost_from_view"))
        timeline.append({"t": round(gone_at, 2), "state": "SUBMERSION",
                         "note": "Possible submersion (lost from view)"})
        state = "SUBMERSION"

    # Rule A: someone we were following vanished in open water and did not come back (they were not in trouble yet,
    # so this is a Level 1 "please check", not an alarm). Never at a wall or where standing is normal.
    last = seq[-1] if seq else {"location": pcfg.get("default_location", "Pool")}
    if (vanished and warn_on and state not in ("DISTRESS", "SUBMERSION") and last.get("in_water", True)
            and not last.get("near_wall") and not any(w in str(last.get("location", "")).lower() for w in standing_words)):
        lost_at = float(vanished["lost_at"])
        alert_at = lost_at + float(pcfg["vanish_lost_s"])
        events.append({"person_id": pid, "event": "early_warning", "start_time": round(lost_at, 2),
                       "alert_time": round(alert_at, 2), "end_time": round(alert_at, 2), "risk_score": round(warn_risk, 2),
                       "evidence": ["vanished_in_water"], "location": last.get("location", "")})
        timeline.append({"t": round(alert_at, 2), "state": "WARNING", "note": "Lost from view in the water: check"})

    # Rule C: vanished in open water and somebody came back up at the very same spot many seconds later.
    if (same_spot and warn_on and state not in ("DISTRESS", "SUBMERSION") and last.get("in_water", True)
            and not last.get("near_wall") and not any(w in str(last.get("location", "")).lower() for w in standing_words)):
        alert = {"person_id": pid, "event": "early_warning", "start_time": round(float(same_spot["lost_at"]), 2),
                 "alert_time": round(float(same_spot["back_at"]), 2), "end_time": round(float(same_spot["back_at"]), 2),
                 "risk_score": round(warn_risk, 2), "evidence": ["back_at_same_spot"],
                 "location": last.get("location", "")}
        if "x" in same_spot:                      # where they vanished (processing px): marked on the video
            alert["spot"] = [same_spot["x"], same_spot["y"]]
        events.append(alert)
        timeline.append({"t": round(float(same_spot["back_at"]), 2), "state": "WARNING",
                         "note": "Seen again at the same spot: may have been under the water"})
    return {"events": events, "risk": risk_curve, "timeline": timeline, "final_state": state}


def _state_for(r: float, pcfg: dict) -> str:
    if r >= float(pcfg["warning_risk"]):
        return "WARNING"
    if r >= float(pcfg["watch_risk"]):
        return "WATCH"
    return "NORMAL"


def _evidence(sig: dict, transition: bool) -> set:
    ev = {k for k in ("vertical_posture", "low_displacement", "repeated_arm_motion", "unstable_head",
                      "stationary_upright", "passive_still") if sig.get(k)}
    if transition and sig["vertical_posture"]:
        ev.add("transition_from_swimming")
    return ev


EVIDENCE_ORDER = ("vertical_posture", "low_displacement", "repeated_arm_motion", "unstable_head",
                  "stationary_upright", "passive_still", "transition_from_swimming", "head_submerged", "lost_from_view",
                  "vanished_in_water", "back_at_same_spot")
EVIDENCE_TEXT = {"vertical_posture": "Vertical posture", "low_displacement": "Little or no forward progress",
                 "repeated_arm_motion": "Repeated distress-like arm movement", "unstable_head": "Unstable head position",
                 "transition_from_swimming": "Was swimming normally just before (distress transition)",
                 "stationary_upright": "Upright and not moving for a long time (not calm treading water)",
                 "passive_still": "Still and face-down for a long time (possible passive drowning)",
                 "head_submerged": "Head went under the water", "lost_from_view": "Lost from view in the water",
                 "vanished_in_water": "Vanished in open water and did not come back into view",
                 "back_at_same_spot": "Out of sight, then seen again at the same spot (may have been under the water)"}


def _ordered(ev: set) -> list[str]:
    return [k for k in EVIDENCE_ORDER if k in ev]


# The four alert kinds, least to most urgent. "under_water" is Rule C (seen again at the very same spot after a long
# disappearance): stored as an early warning with that evidence, shown and spoken as its own urgent kind.
ALERT_KINDS = ("aquatic_warning", "under_water", "aquatic_distress", "submersion")


def alert_kind(alert: dict) -> str:
    """The kind of one alert: a stage-3 alert (event + evidence) or a standard event (behavior + metrics.evidence_keys).
    Derived from the stored fields, so results saved before this kind existed show it too."""
    evidence = alert.get("evidence") if isinstance(alert.get("evidence"), list) else \
        (alert.get("metrics") or {}).get("evidence_keys") or []
    if "back_at_same_spot" in evidence:
        return "under_water"
    kind = alert.get("behavior") or alert.get("event")
    return {"possible_submersion": "submersion", "early_warning": "aquatic_warning",
            "high_risk_aquatic_distress": "aquatic_distress"}.get(kind, kind)


def picture_area(x: float, y: float, width: float, height: float) -> str:
    """Which ninth of the picture a point is in, in words ("top left", "centre", "bottom right")."""
    if not width or not height:
        return ""
    col = ("left", "", "right")[min(2, max(0, int(3 * x / width)))]
    row = ("top", "", "bottom")[min(2, max(0, int(3 * y / height)))]
    return f"{row} {col}".strip() or "centre"


def where_text(alert: dict, spoken: bool = False) -> str:
    """Where an alert is, for a lifeguard: camera name, zone and part of the picture (the parts that are known).
    spoken=True gives a sentence for the voice ("Camera CAM03 Far side, in the Pool, top left of the picture")."""
    m = alert.get("metrics") or {}
    camera = str(alert.get("camera") or m.get("camera") or "").strip()
    zone = str(alert.get("location") or m.get("location") or alert.get("zone") or "").strip()
    area = str(alert.get("area") or m.get("area") or "").strip()
    if not spoken:
        return " · ".join(p for p in (camera, zone, f"{area} of the picture" if area else "") if p) or "Pool"
    parts = ([f"Camera {camera}"] if camera else []) + ([f"in the {zone}"] if zone else []) + \
        ([f"{area} of the picture"] if area else [])
    text = ", ".join(parts) or "in the pool"
    return text[0].upper() + text[1:]


def never_judged(dash: dict) -> list[int]:
    """IDs of the tracked people the rules never scored in the whole video (no risk reading at all: seen too briefly,
    too small, or outside the water). 'No alert' says nothing about them, so the all-clear must name them."""
    risk = dash.get("risk") or {}
    return [int(p["person_id"]) for p in dash.get("people") or [] if not risk.get(str(p["person_id"]))]


def all_clear_text(dash: dict) -> str:
    """The no-alert sentence, honest about the people the rules could not judge."""
    total = len(dash.get("people") or [])
    unjudged = len(never_judged(dash))
    if not unjudged:
        return "No distress: every swimmer stayed within normal behaviour."
    return (f"No alert among the {total - unjudged} of {total} people the rules could judge. "
            f"{unjudged} of {total} were not judged (seen too briefly, too small or outside the water): "
            f"check them by eye.")


def _risk_score(risks: list[float], held_s: float, hold: float) -> float:
    """Mean risk of the run, nudged up the longer it lasts (max 1.0)."""
    mean = sum(risks) / max(1, len(risks))
    return round(min(1.0, mean * (0.9 + 0.1 * min(1.0, held_s / (2 * hold)))), 2)


def _submersion(pid: int, t: float, location: str, why: str) -> dict:
    return {"person_id": pid, "event": "possible_submersion", "start_time": round(float(t), 2),
            "alert_time": round(float(t), 2), "end_time": round(float(t), 2), "risk_score": 0.98,
            "evidence": [why], "location": location}


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: distress reasoning from stage-2 behaviour samples")
    ap.add_argument("--behaviour", required=True, help="stage2_behaviour.jsonl")
    ap.add_argument("--scenario", default="pool")
    ap.add_argument("--out", default="stage3_events.json")
    ap.add_argument("--gaps", default=None,
                    help="stage2_gaps.json (gaps inside joined tracks); default: the one next to --behaviour, if any")
    args = ap.parse_args(argv)
    cfg = utils.load_config(scenario=args.scenario)
    with open(args.behaviour, encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]
    gaps_path = Path(args.gaps) if args.gaps else Path(args.behaviour).with_name("stage2_gaps.json")
    gaps = ({int(k): [tuple(g) for g in v] for k, v in utils.read_json(gaps_path).items()}
            if gaps_path.is_file() else None)
    res = reason(samples, cfg, gaps=gaps)
    utils.write_json(args.out, {"events": res["events"], "timeline": res["timeline"]})
    for e in res["events"]:
        print(f"Person #{e['person_id']}: {e['event']} at {utils.fmt_time(e['alert_time'])} "
              f"(risk {e['risk_score']:.0%}) evidence: {', '.join(e['evidence'])}")
    print(f"{len(res['events'])} event(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
