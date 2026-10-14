"""Incident chains: rate every event (severity) and link one entity's events into a story.

Pure Python (no OpenCV, no NumPy work). This is where a list of separate events such as

    Person #3 loitering 00:12-00:30   ->   Person #3 inside the zone 00:31-00:36

becomes ONE incident with a title ("Waited nearby, then entered the restricted zone"), a severity
and a plain-English story with the gaps between the steps. See docs/DESIGN.md section 4.1 and 4.8.

Entry point: build_chains(events, cfg) -> list of chain dicts. It also sets event["severity"]
("low" | "medium" | "high") on every event, even when chaining is switched off.
"""
from __future__ import annotations

import utils

LEVELS = ("low", "medium", "high")

# Known two-step patterns, in priority order (the most alarming first). A chain gets the title of the
# highest-priority pattern found among its consecutive behaviours.
PATTERN_TITLES = (
    (("aquatic_distress", "submersion"), "Distress in the water, then went under"),
    (("aquatic_warning", "submersion"), "Warning signs in the water, then went under"),
    (("aquatic_warning", "aquatic_distress"), "Warning signs, then distress in the water"),
    (("zone_intrusion", "near_miss"), "Entered the danger zone and nearly got hit"),
    (("running", "near_miss"), "Ran and nearly collided"),
    (("near_miss", "running"), "Near miss, then fled"),
    (("loitering", "zone_intrusion"), "Waited nearby, then entered the restricted zone"),
    (("running", "zone_intrusion"), "Rushed into the restricted zone"),
    (("zone_intrusion", "running"), "Ran away after entering the restricted zone"),
    (("loitering", "running"), "Waited, then suddenly ran"),
)
DEFAULT_TITLE = "Repeated suspicious activity"

# Behaviours that happen at the same time (overlapping events) and what they mean together.
OVERLAP_TITLES = (
    (("near_miss", "zone_intrusion"), "Nearly hit inside the danger zone"),
    (("fall", "zone_intrusion"), "Fell inside the restricted zone"),
    (("loitering", "zone_intrusion"), "Lingered inside the restricted zone"),
    (("running", "zone_intrusion"), "Ran through the restricted zone"),
)

_EPS = 1e-9


# --------------------------------------------------------------------------- severity

def raise_level(severity: str, steps: int = 1) -> str:
    """Move a severity up by `steps` levels (low -> medium -> high); high stays high."""
    index = LEVELS.index(severity) if severity in LEVELS else 0
    return LEVELS[min(len(LEVELS) - 1, index + steps)]


def max_level(a: str | None, b: str | None) -> str:
    """The more serious of two severities (None counts as low)."""
    ranks = [LEVELS.index(x) if x in LEVELS else 0 for x in (a, b)]
    return LEVELS[max(ranks)]


def _is_unusual(event: dict, cfg: dict) -> bool:
    """Does the event's scene-baseline note say this entity stood out from the rest of the scene?"""
    base = event.get("baseline")
    if not base:
        return False
    if base.get("unusual") is not None:
        return bool(base["unusual"])
    threshold = float((((cfg or {}).get("behaviors") or {}).get("baseline") or {}).get("z_threshold", 3.0))
    score = base.get("anomaly_score")
    return score is not None and float(score) >= threshold


def event_severity(event: dict, cfg: dict) -> str:
    """Severity of one event on its own (before chaining).

    zone_intrusion = medium; running = medium if confidence >= 0.6 else low; loitering = low;
    fall and near_miss are always high; crowding is medium, high at twice the crowd limit.
    An entity that is unusual compared with the scene is raised one level (fall / near_miss are
    already at the top).
    """
    behavior = event.get("behavior")
    metrics = event.get("metrics") or {}
    if behavior in ("fall", "near_miss", "aquatic_distress", "submersion"):
        return "high"
    if behavior == "aquatic_warning":             # early warning "may need help": look now, not yet an alarm
        # ...except Rule C "possibly under water" (seen again at the same spot): its own urgent kind (G4)
        return "high" if "back_at_same_spot" in (metrics.get("evidence_keys") or []) else "medium"
    if behavior == "approaching":                 # about a second to contact = high, else medium
        return "high" if (metrics.get("min_ttc_s") is not None and metrics["min_ttc_s"] <= 1.0) else "medium"
    if behavior == "zone_intrusion":
        level = "medium"
    elif behavior == "running":
        level = "medium" if float(event.get("confidence") or 0.0) >= 0.6 else "low"
    elif behavior == "crowding":
        crowd_cfg = ((cfg or {}).get("behaviors") or {}).get("crowding") or {}
        need = metrics.get("min_count") or crowd_cfg.get("min_count") or 5
        level = "high" if float(metrics.get("max_count") or 0) >= 2 * float(need) else "medium"
    else:                                   # loitering and anything unknown
        level = "low"
    return raise_level(level) if _is_unusual(event, cfg) else level


# --------------------------------------------------------------------------- chains

def build_chains(events: list[dict], cfg: dict) -> list[dict]:
    """Rate every event and link each entity's consecutive events into incident chains.

    * Sets event["severity"] on every event (in place).
    * A chain = the same entity_id with >= 2 events in start order, where each next event starts
      at most chains.max_gap_s after everything before it has ended. Events with entity_id None
      (crowding) are never chained.
    * Chain severity is high if it holds a zone_intrusion plus another behaviour (and never lower
      than its most serious event, so a chain with a fall or near miss is high); every event in
      the chain is raised to at least that severity.
    Returns the chains ordered by start time, numbered from 1.
    """
    for ev in events:
        ev["severity"] = event_severity(ev, cfg)

    chain_cfg = (cfg or {}).get("chains") or {}
    if not chain_cfg.get("enabled", True):
        return []
    max_gap = float(chain_cfg.get("max_gap_s", 15.0))

    by_entity: dict = {}
    for ev in events:
        if ev.get("entity_id") is not None:
            by_entity.setdefault(ev["entity_id"], []).append(ev)

    runs: list[list[dict]] = []
    for evs in by_entity.values():
        evs.sort(key=lambda e: (e["start_s"], e["end_s"], e.get("event_id") or 0))
        run, run_end = [evs[0]], evs[0]["end_s"]
        for ev in evs[1:]:
            if ev["start_s"] - run_end <= max_gap + _EPS:       # still the same story
                run.append(ev)
                run_end = max(run_end, ev["end_s"])
            else:                                               # too long a pause: start a new run
                runs.append(run)
                run, run_end = [ev], ev["end_s"]
        runs.append(run)

    runs = [r for r in runs if len(r) >= 2]
    runs.sort(key=lambda r: (r[0]["start_s"], str(r[0]["entity_id"])))
    return [_make_chain(n, run, cfg) for n, run in enumerate(runs, start=1)]


def _make_chain(chain_id: int, run: list[dict], cfg: dict) -> dict:
    """Build one chain dict from a run of one entity's events and raise its events to its severity."""
    behaviors = [e["behavior"] for e in run]
    severity = "high" if ("zone_intrusion" in behaviors and len(set(behaviors)) >= 2) else "medium"
    for ev in run:
        severity = max_level(severity, ev.get("severity"))
    for ev in run:
        ev["severity"] = max_level(ev.get("severity"), severity)

    start_s = min(e["start_s"] for e in run)
    end_s = max(e["end_s"] for e in run)
    entity_id = run[0]["entity_id"]
    return {
        "chain_id": chain_id,
        "entity_id": entity_id,
        "entity_name": run[0].get("entity_name") or utils.entity_name(cfg, entity_id),
        "event_ids": [e.get("event_id") for e in run],
        "pattern": " -> ".join(behaviors),
        "title": chain_title(behaviors, run),
        "start_s": start_s,
        "end_s": end_s,
        "start": utils.fmt_time(start_s),
        "end": utils.fmt_time(end_s),
        "severity": severity,
        "story": chain_story(run, cfg),
    }


def chain_title(behaviors: list[str], run: list[dict] | None = None) -> str:
    """Human title for a run of one entity's events.

    Events that happen AT THE SAME TIME are described together first (e.g. loitering while
    inside the zone = "Lingered inside the restricted zone"); otherwise the first known
    two-step pattern in time order wins, else the default title.
    """
    if run:
        for (a, b), title in OVERLAP_TITLES:
            evs_a = [e for e in run if e["behavior"] == a]
            evs_b = [e for e in run if e["behavior"] == b]
            if any(_overlap(x, y) for x in evs_a for y in evs_b):
                return title
    pairs = set(zip(behaviors, behaviors[1:]))
    for pattern, title in PATTERN_TITLES:
        if pattern in pairs:
            return title
    if len(set(behaviors)) == 1:
        return f"Repeated {utils.DEFAULT_BEHAVIOR_NAMES.get(behaviors[0], behaviors[0]).lower()}"
    return DEFAULT_TITLE


def _overlap(a: dict, b: dict, min_overlap_s: float = 1.0) -> bool:
    """True if two events share at least `min_overlap_s` seconds."""
    return min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]) >= min_overlap_s


# --------------------------------------------------------------------------- story

def _secs(value) -> str:
    """Seconds for a sentence: '5', '1.2', '25' (whole numbers from 10 s up)."""
    value = float(value)
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}".rstrip("0").rstrip(".")


def _span(ev: dict) -> str:
    """'00:12-00:30' for an event ('00:03.2-00:03.8' when both ends fall in the same second)."""
    start = ev.get("start") or utils.fmt_time(ev["start_s"])
    end = ev.get("end") or utils.fmt_time(ev["end_s"])
    if start == end:
        start, end = utils.fmt_time_precise(ev["start_s"]), utils.fmt_time_precise(ev["end_s"])
    return f"{start}-{end}"


def _clause(ev: dict, cfg: dict) -> tuple[str, str]:
    """(what happened, what happened next/how long) for one event, as plain English.

    Scenarios that rename a behaviour (e.g. zone_intrusion -> "Left the pen") use that name instead
    of the default verb, so the story always speaks the scenario's language.
    """
    behavior = ev["behavior"]
    metrics = ev.get("metrics") or {}
    start = ev.get("start") or utils.fmt_time(ev["start_s"])
    duration = ev.get("duration_s") if ev.get("duration_s") is not None else ev["end_s"] - ev["start_s"]
    name = utils.behavior_name(cfg, behavior)
    if name != utils.DEFAULT_BEHAVIOR_NAMES.get(behavior):          # scenario wording
        return f"{name[:1].lower()}{name[1:]} {_span(ev)}", ""

    if behavior == "loitering":
        return f"loitered {_span(ev)}", ""
    if behavior == "zone_intrusion":
        zone = ev.get("zone") or metrics.get("zone") or "restricted"
        return f"entered zone '{zone}' at {start}", f"stayed {_secs(duration)} s"
    if behavior == "running":
        return f"ran {_span(ev)}", ""
    if behavior == "fall":
        if metrics.get("fell"):
            return f"fell at {start}", f"stayed down {_secs(duration)} s"
        return f"was lying down {_span(ev)}", ""
    if behavior == "near_miss":
        other = utils.entity_name(cfg, ev.get("other_entity"), metrics.get("other_class")) \
            if ev.get("other_entity") is not None else "someone"
        moment = utils.fmt_time(metrics.get("peak_time_s", ev["start_s"]))
        contact = " (possible contact)" if metrics.get("contact") else ""
        return f"nearly collided with {other} at {moment}{contact}", ""
    return f"{name[:1].lower()}{name[1:]} {_span(ev)}", ""


def _gap_text(gap: float) -> str:
    """How long after the previous step: '(3 s later)', '(right after)' or '(at the same time)'."""
    if gap < 0:
        return "(while doing so)"
    if gap < 0.5:
        return "(right after)"
    return f"({_secs(gap)} s later)"


def chain_story(run: list[dict], cfg: dict) -> str:
    """One sentence per chain, built from its events in order, with the gaps between them.

    Example: "Person #3 loitered 00:12-00:30, then entered zone 'restricted' at 00:31 (1 s later)
    and stayed 5 s."
    """
    first = run[0]
    who = first.get("entity_name") or utils.entity_name(cfg, first.get("entity_id"))
    parts = []
    covered_until = first["end_s"]
    for i, ev in enumerate(run):
        head, tail = _clause(ev, cfg)
        text = f"{who} {head}" if i == 0 else f"then {head} {_gap_text(ev['start_s'] - covered_until)}"
        if tail:
            text += f" and {tail}"
        parts.append(text)
        covered_until = max(covered_until, ev["end_s"])
    return ", ".join(parts) + "."
