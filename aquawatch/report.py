"""Final outputs: events.json, summary.txt and the self-contained report.html.

No OpenCV here (images are only read from disk and embedded as base64), so the report can
be regenerated anywhere. Everything shown is taken from the events / incidents / tracks / config,
so the numbers on the page are exactly the numbers the detector used.

Entry point: write_outputs(final, meta, tracks_summary, out_dir).

`final` is the dict from events.build_events plus, when the later steps ran, "incidents" (the chains
from chains.build_chains) and "highlight_reel" (file name of highlights.mp4).

`meta` is a plain dict from run.py. Keys used if present (all optional): video, duration_s, fps,
stride, proc_size, orig_size, device, model, n_people, zones (list of names or zone dicts),
pose (bool), elapsed_s, scenario ({name, title}), privacy (bool), processing_fps (float or None),
config (the cfg dict: used for scenario wording and the settings table).
All user-facing names come from utils.entity_name / behavior_name / unit_name, so a scenario
preset (livestock, traffic, ...) changes the wording of the whole report.
"""
from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime
from pathlib import Path

import utils
from utils import BEHAVIOR_COLORS_HEX, ensure_dir, fmt_time, fmt_time_precise, write_json

_e = html.escape  # every piece of text that reaches the HTML goes through this

SEVERITY_COLORS = {"high": "#d64545", "medium": "#d99a00", "low": "#4f7fb8"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
PRIVACY_NOTE = "Privacy mode: faces blurred in all outputs"


# =========================================================================== small helpers

def _num(x, nd: int = 2, default: str = "n/a") -> str:
    """Format a number for display; None / NaN become 'n/a'."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else f"{v:.{nd}f}"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


def _entity_plural(cfg: dict, n: int) -> str:
    """'1 person', '4 people', '3 animals': the scenario's entity word, counted."""
    word = utils.entity_word(cfg).lower()
    if word == "person" and len(((cfg or {}).get("model") or {}).get("classes") or [0]) > 1:
        word = "object"                       # people AND vehicles are tracked (same rule as run.py)
    return _plural(n, word, "people" if word == "person" else word + "s")


def _entity_plural_word(cfg: dict) -> str:
    """Just the plural noun: 'people', 'animals'."""
    return _entity_plural(cfg, 2).split(" ", 1)[1]


def _video_name(meta: dict) -> str:
    video = meta.get("video")
    return Path(str(video)).name if video else "video"


def _jsonable(obj):
    """Make meta safe for json: unknown objects become strings, large detection lists are dropped."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if k != "detections"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "item"):  # NumPy scalar
        return obj.item()
    return str(obj)


def _data_uri(path: Path) -> str | None:
    """Read an image and return it as a data: URI, or None if it is missing / unreadable."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _img(out_dir: Path, rel, alt: str) -> str:
    """<img> tag with the file embedded; empty string when the image does not exist."""
    if not rel:
        return ""
    uri = _data_uri(Path(out_dir) / str(rel))
    return f'<img src="{uri}" alt="{_e(alt)}">' if uri else ""


def _resolve_cfg(meta: dict) -> dict:
    """The config the run used: meta['config'] if given, else the scenario preset, else the defaults."""
    cfg = meta.get("config") or meta.get("cfg")
    if isinstance(cfg, dict):
        return cfg
    scen = meta.get("scenario")
    name = scen.get("name") if isinstance(scen, dict) else scen
    try:
        if name and name in utils.list_scenarios():
            return utils.load_config(scenario=name)
    except Exception:           # a broken preset must never stop the report
        pass
    return utils.load_config()


def _scenario_title(meta: dict, cfg: dict) -> str:
    scen = meta.get("scenario")
    if isinstance(scen, dict) and scen.get("title"):
        return str(scen["title"])
    return str((cfg.get("scenario") or {}).get("title") or "")


def _behavior_text(ev_or_key, cfg: dict) -> str:
    """Display name of a behaviour in the scenario's words ('Zone intrusion', 'Near miss')."""
    if isinstance(ev_or_key, dict):
        return str(ev_or_key.get("behavior_name") or utils.behavior_name(cfg, ev_or_key.get("behavior", "")))
    return str(utils.behavior_name(cfg, str(ev_or_key)))


def _behavior_label(ev_or_key, cfg: dict) -> str:
    """Upper-case badge text of a behaviour ('ZONE INTRUSION', 'NEAR MISS')."""
    return _behavior_text(ev_or_key, cfg).upper()


def _who(ev: dict, cfg: dict) -> str:
    """Who an event is about: 'Person #3', 'Group of 6', or both parties of a near miss."""
    entities = ev.get("entities") or []
    entity_id = ev.get("entity_id")
    if entity_id is None:                                       # crowding: a whole group
        return ev.get("entity_name") or f"Group of {len(entities)}"
    first = ev.get("entity_name") or utils.entity_name(cfg, entity_id)
    other = ev.get("other_entity")
    if other is None or f"#{other}" in first:
        return first
    other_cls = (ev.get("metrics") or {}).get("other_class")
    return f"{first} and {utils.entity_name(cfg, other, other_cls)}"


def _involves(ev: dict, entity_id) -> bool:
    """Is this entity part of the event (as the main entity, the other party or a group member)?"""
    return entity_id is not None and (ev.get("entity_id") == entity_id or entity_id in (ev.get("entities") or []))


def _chain_lookup(incidents: list[dict]) -> dict:
    """event_id -> chain dict, for events that belong to an incident."""
    return {eid: c for c in incidents for eid in c.get("event_ids", [])}


def _severity_key(item_severity, start_s) -> tuple:
    return SEVERITY_ORDER.get(item_severity, 3), start_s


# =========================================================================== guard summary

def _event_detail(ev: dict, cfg: dict) -> str:
    """Short plain-English description of what an event measured (no names, no times)."""
    m = ev.get("metrics") or {}
    unit = utils.unit_name(cfg)
    dur = float(ev.get("duration_s") or 0.0)
    beh = ev.get("behavior")
    if beh == "loitering":
        return f"stayed near one spot for {dur:.0f} s"
    if beh == "zone_intrusion":
        return f"inside zone '{ev.get('zone') or m.get('zone') or 'zone'}' for {dur:.1f} s"
    if beh == "running":
        if m.get("mean_speed_bh_s") is None:   # metrics missing: still say something useful
            return f"moving fast for {dur:.1f} s"
        return f"moved at {_num(m.get('mean_speed_bh_s'), 1)} {unit}/s (peak {_num(m.get('peak_speed_bh_s'), 1)})"
    if beh == "fall":
        return f"fell and stayed down {dur:.0f} s" if m.get("fell") else f"lying down for {dur:.0f} s"
    if beh == "aquatic_distress":
        return f"distress pattern in the {m.get('location') or 'pool'}, risk {_num(100 * float(m.get('risk_score') or 0), 0)}%"
    if beh == "submersion":
        return f"possibly went under in the {m.get('location') or 'pool'}: check now"
    if beh == "aquatic_warning" and "back_at_same_spot" in (m.get("evidence_keys") or []):
        where = str(m.get("where") or m.get("location") or "pool").replace(" · ", ", ")    # ASCII for summary.txt
        return f"out of sight, then seen again at the same spot ({where}): go and check now"
    if beh == "aquatic_warning":
        return (f"warning signs in the {m.get('location') or 'pool'}: may need help, "
                f"risk {_num(100 * float(m.get('risk_score') or 0), 0)}%")
    if beh == "crowding":
        return (f"{_num(m.get('max_count'), 0)} {_entity_plural_word(cfg)} at once "
                f"(avg {_num(m.get('mean_count'), 1)}) in '{ev.get('zone') or m.get('zone') or 'the area'}' for {dur:.0f} s")
    if beh == "near_miss":
        text = f"came within {_num(m.get('min_distance_bh'), 2)} {unit}"
        if m.get("max_closing_speed_bh_s") is not None:
            text += f", closing at {_num(m.get('max_closing_speed_bh_s'), 1)} {unit}/s"
        if m.get("contact"):
            return text + " (POSSIBLE CONTACT)"
        if m.get("min_ttc_s") is not None:
            text += f" (time-to-collision {_num(m.get('min_ttc_s'), 1)} s)"
        return text
    return f"for {dur:.1f} s"


def _event_phrase(ev: dict, cfg: dict) -> str:
    """One plain-English clause describing an event for the guard summary."""
    return f"{_who(ev, cfg)}: {_behavior_text(ev, cfg)} - {_event_detail(ev, cfg)}"


def _sev_tag(severity) -> str:
    return f" [{str(severity).upper()}]" if severity else ""


def make_summary(final: dict, meta: dict, cfg: dict | None = None) -> str:
    """3-6 line plain-text summary for a security guard.

    Incidents (linked chains) come first, then the events that are not part of a chain, most
    serious first. Ends with the things noticed but not flagged, and the privacy note if on.
    """
    events = final.get("events", []) or []
    incidents = final.get("incidents", []) or []
    meta = meta or {}
    cfg = cfg if isinstance(cfg, dict) else _resolve_cfg(meta)
    n_people = meta.get("n_people")
    if n_people is None:  # fall back to what the result itself knows
        n_people = (final.get("baseline") or {}).get("n_tracks")
    if n_people is None:
        n_people = len({i for e in events for i in (e.get("entities") or [e.get("entity_id")]) if i is not None})
    n_people = int(n_people)
    dur = meta.get("duration_s")
    head = f"Video {_video_name(meta)}" + (f" ({fmt_time(float(dur))})" if dur else "")
    scen = meta.get("scenario") if isinstance(meta.get("scenario"), dict) else {}
    if scen.get("name") and scen.get("name") != "campus" and scen.get("title"):
        head += f" [{scen['title']}]"

    # things noticed but not flagged, for the last line
    flagged = {i for e in events for i in ((e.get("entities") or []) + [e.get("entity_id")]) if i is not None}
    n_almost = len(final.get("near_misses", []) or [])
    n_unusual = len([u for u in (final.get("unusual_tracks", []) or []) if u.get("entity_id") not in flagged])
    notes = []
    if n_almost:
        notes.append(_plural(n_almost, "almost-flagged case") + " just under a rule")
    if n_unusual:
        notes.append(_entity_plural(cfg, n_unusual) + " unusual compared with the scene's typical behaviour")
    privacy = f" {PRIVACY_NOTE}." if meta.get("privacy") else ""
    feed = meta.get("feed_health") or {}
    feed_line = [f"{feed['summary']}." if not str(feed['summary']).endswith(".") else feed["summary"]] \
        if feed.get("problems") else []            # a broken feed comes right after the title, before any all-clear
    if feed.get("warnings"):                       # camera moving / top-down view (G7): results less reliable
        feed_line.append("CAMERA VIEW WARNING (results less reliable): "
                         + "; ".join(w["text"] for w in feed["warnings"]) + ".")

    if not events:
        lines = [f"{head}."] + feed_line + [f"No incidents. {_entity_plural(cfg, n_people)} seen, "
                 "all behaviour within normal range."]
        if notes:
            lines.append("Also noted (below the rules, not flagged): " + " and ".join(notes) + ".")
        if privacy:
            lines.append(privacy.strip())
        return "\n".join(lines)

    in_chain = {eid for c in incidents for eid in c.get("event_ids", [])}
    loose = [e for e in events if e.get("event_id") not in in_chain]
    items = [("chain", c) for c in sorted(incidents, key=lambda c: _severity_key(c.get("severity"), c["start_s"]))]
    items += [("event", e) for e in sorted(loose, key=lambda e: _severity_key(e.get("severity"), e["start_s"]))]

    lines = [f"{head}, {_entity_plural(cfg, n_people)} seen. {_plural(len(items), 'incident')}:"]
    if feed_line:                                  # the broken feed first, then the incidents
        lines = [f"{head}, {_entity_plural(cfg, n_people)} seen."] + feed_line + \
            [f"{_plural(len(items), 'incident')}:"]
    shown = items if len(items) <= 4 else items[:3]   # keep the summary short for a guard
    for kind, item in shown:
        when = f"{item.get('start') or fmt_time(item['start_s'])}-{item.get('end') or fmt_time(item['end_s'])}"
        if kind == "chain":
            who = item.get("entity_name") or utils.entity_name(cfg, item.get("entity_id"))
            lines.append(f"- {when} {who}: {item.get('title')}{_sev_tag(item.get('severity'))}")
        else:
            verdict = " [pose check disagrees]" if item.get("verified") is False else ""
            lines.append(f"- {when} {_event_phrase(item, cfg)}{_sev_tag(item.get('severity'))}{verdict}")
    if len(items) > len(shown):
        lines.append(f"- ...and {len(items) - len(shown)} more (see report.html).")
    last = ("Also noted (not flagged): " + " and ".join(notes) + "." if notes else "No other unusual activity.")
    lines.append(last + privacy)
    return "\n".join(lines)


# =========================================================================== HTML building blocks

_CSS = """
:root{--bg:#f3f5f8;--card:#ffffff;--ink:#1b2430;--muted:#5d6978;--line:#dfe4ea;--accent:#2a56c6;
--good:#2e9d57;--warn:#d99a00;--bad:#d64545;--soft:#eef2f8;--crim:#dc143c}
*{box-sizing:border-box}
html{color-scheme:light}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
header.top{background:var(--card);border:1px solid var(--line);border-top:5px solid var(--accent);
border-radius:12px;padding:22px 24px;margin-bottom:18px}
header.top h1{margin:0 0 2px;font-size:26px;letter-spacing:-.01em}
header.top .scen{margin:0 0 2px;font-weight:650;color:var(--accent)}
header.top .sub{color:var(--muted);margin:0 0 4px}
header.top .speed{color:var(--muted);margin:0 0 16px;font-size:13.5px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--soft);border-radius:10px;padding:10px 14px}
.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.tile .v{font-size:22px;font-weight:650;word-break:break-word}
.tile .v.small{font-size:16px}
.tile.alert{background:#fdeceb}.tile.alert .v{color:var(--bad)}
.tile.ok{background:#e8f6ee}.tile.ok .v{color:var(--good)}
.reel{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;justify-content:space-between;
background:linear-gradient(100deg,#1d3a8a,#2a56c6);color:#fff;border-radius:12px;padding:16px 22px;margin-bottom:18px}
.reel strong{display:block;font-size:18px}.reel span{font-size:13.5px;opacity:.9}
a.btn{display:inline-block;background:#fff;color:#1d3a8a;font-weight:750;text-decoration:none;
padding:10px 20px;border-radius:999px;border:2px solid #fff}
a.btn:hover{background:#e8eefc}
.privacy{background:#eef2f8;border:1px solid var(--line);border-left:5px solid var(--accent);border-radius:10px;
padding:10px 16px;margin-bottom:18px;font-size:14px}
.privacy b{color:var(--accent)}
section{margin-top:26px}
h2{font-size:19px;margin:0 0 4px}
.lead{color:var(--muted);margin:0 0 12px;font-size:14px}
.summary{background:#fffaf0;border:1px solid #f0dfb5;border-left:5px solid var(--warn);border-radius:10px;
padding:14px 18px}
.summary.clear{background:#eef9f2;border-color:#bfe3cd;border-left-color:var(--good)}
.summary p{margin:2px 0}.summary p.first{font-weight:650}
.summary ul{margin:6px 0 6px 18px;padding:0}.summary li{margin:3px 0}
.card,.inc{--c:#888;background:var(--card);border:1px solid var(--line);border-left:6px solid var(--c);
border-radius:12px;padding:16px 18px;margin:14px 0;break-inside:avoid;page-break-inside:avoid;scroll-margin-top:12px}
.inc .story{font-size:16px;margin:6px 0 10px}
.steps{display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;font-size:13px;color:var(--muted)}
a.step{--c:#888;display:inline-block;text-decoration:none;color:var(--ink);background:var(--soft);
border:1px solid var(--line);border-left:5px solid var(--c);border-radius:8px;padding:3px 10px}
a.step:hover{background:#e3eaf7}
.card .head,.inc .head{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;margin-bottom:10px}
.card h3,.inc h3{margin:0;font-size:17px}
.badge{display:inline-block;background:var(--c);color:var(--bc,#fff);font-weight:700;font-size:12px;
letter-spacing:.06em;padding:3px 10px;border-radius:999px}
.badge.contact{background:#fff;color:var(--crim);border:2px solid var(--crim);padding:1px 9px}
.partof{font-size:13px;color:var(--muted);margin:-2px 0 8px}
.when{margin-left:auto;color:var(--muted);font-variant-numeric:tabular-nums}
.media{display:grid;grid-template-columns:3fr 2fr;gap:12px;align-items:start}
.media figure{margin:0}
.media img{width:100%;height:auto;display:block;border-radius:8px;border:1px solid var(--line)}
.media figcaption{font-size:12px;color:var(--muted);margin-top:3px}
.cols{display:grid;grid-template-columns:1.2fr 1fr;gap:6px 22px;margin-top:12px}
.evidence{font-size:16px;margin:0 0 8px}
.why{font-size:13px;color:var(--muted);margin:2px 0}
.lab{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:10px 0 4px}
.conf{display:flex;align-items:center;gap:10px}
.conf .n{font-size:22px;font-weight:700;min-width:3.2em;font-variant-numeric:tabular-nums}
.bar{flex:1;height:12px;background:#e7ebf0;border-radius:999px;overflow:hidden}
.bar>span{display:block;height:100%;border-radius:999px}
.parts{display:grid;grid-template-columns:auto 1fr auto;gap:3px 8px;align-items:center;margin-top:6px;
font-size:12.5px;color:var(--muted)}
.parts .bar{height:7px}
.parts b{color:var(--ink);font-variant-numeric:tabular-nums}
.chip{display:inline-block;font-size:12.5px;padding:2px 9px;border-radius:8px;margin-right:6px;font-weight:600}
.chip.good{background:#e1f4e9;color:#1d7a43}.chip.bad{background:#fbe3e3;color:#a82a2a}
.chip.unsure{background:#fdf1d6;color:#8a6200}.chip.info{background:var(--soft);color:var(--accent)}
.note{font-size:13.5px;margin:4px 0}
dl.metrics{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0;font-size:13px}
dl.metrics dt{color:var(--muted)}dl.metrics dd{margin:0;font-variant-numeric:tabular-nums}
.pair{display:grid;grid-template-columns:1fr;gap:14px}
.pair img,.full img{width:100%;height:auto;border-radius:10px;border:1px solid var(--line);background:#fff}
.heat{max-width:640px}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{background:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.empty{color:var(--muted);background:var(--card);border:1px dashed var(--line);border-radius:10px;padding:12px 16px}
footer{margin-top:30px;color:var(--muted);font-size:12.5px;border-top:1px solid var(--line);padding-top:12px}
a{color:var(--accent)}
@media (max-width:760px){.media,.cols{grid-template-columns:1fr}.when{margin-left:0}
.card,.inc{padding:14px}th,td{padding:7px 8px}}
@media print{
@page{size:A4;margin:12mm}
body{background:#fff;font-size:12px;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.wrap{max-width:none;padding:0}.noprint{display:none!important}
header.top,.card,.inc,table,.summary{box-shadow:none}
.card,.inc,table,tr,figure{break-inside:avoid;page-break-inside:avoid}
section{margin-top:16px}h2{break-after:avoid}
.media{grid-template-columns:3fr 2fr}.cols{grid-template-columns:1.2fr 1fr}
}
"""

# Metric labels: key -> (label, unit, decimals). Unit "U" = the scenario's size unit (body-heights ...).
_METRIC_LABELS = {
    "duration_s": ("Duration", "s", 1), "min_duration_s": ("Minimum duration (rule)", "s", 0),
    "max_radius_bh": ("Furthest from the spot", "U", 2), "radius_threshold_bh": ("Radius (rule)", "U", 2),
    "mean_speed_bh_s": ("Mean speed", "U/s", 2), "peak_speed_bh_s": ("Peak speed", "U/s", 2),
    "start_threshold_bh_s": ("Running starts above (rule)", "U/s", 2),
    "end_threshold_bh_s": ("Running ends below (rule)", "U/s", 2),
    "max_depth_bh": ("Deepest inside the zone", "U", 2), "mean_det_conf": ("Mean detection confidence", "", 2),
    # fall
    "max_aspect": ("Widest box (width / height)", "", 2),
    "upright_aspect_before": ("Box before the fall (width / height)", "", 2),
    "transition_s": ("Fall window (rule)", "s", 1), "min_down_s": ("Minimum time down (rule)", "s", 0),
    # crowding
    "max_count": ("Most at once", "", 0), "mean_count": ("Average count", "", 1),
    "min_count": ("Crowd limit (rule)", "", 0),
    # near miss
    "min_distance_bh": ("Closest distance", "U", 2), "max_closing_speed_bh_s": ("Fastest closing speed", "U/s", 2),
    "min_ttc_s": ("Shortest time to collision", "s", 2),
    "rel_speed_at_closest_bh_s": ("Relative speed at closest", "U/s", 2), "approach_s": ("Approached for", "s", 1),
}
_PART_LABELS = {"margin": ("Margin past the rule", "margin"), "duration": ("Duration vs minimum", "duration"),
                "detection": ("Detection quality", "detection")}


def _conf_color(c: float) -> str:
    return "#d64545" if c < 0.5 else ("#d99a00" if c < 0.75 else "#2e9d57")


def _badge_text_color(hex_color: str) -> str:
    """Dark text on light badge colours (e.g. orange), white otherwise."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#1b2430" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def _bar(frac: float, color: str) -> str:
    return f'<div class="bar"><span style="width:{max(0.0, min(1.0, frac)) * 100:.0f}%;background:{color}"></span></div>'


def _sev_badge(severity, prefix: str = "") -> str:
    """Coloured severity badge ('HIGH SEVERITY'); empty when the event has no severity."""
    if severity not in SEVERITY_COLORS:
        return ""
    color = SEVERITY_COLORS[severity]
    return (f'<span class="badge" title="Severity" style="--c:{color};--bc:{_badge_text_color(color)}">'
            f'{_e(prefix + severity.upper())} SEVERITY</span>')


def _pose_html(ev: dict) -> str:
    verified = ev.get("verified")
    if verified is None:
        return ""
    pose = ev.get("pose") or {}
    chip = {True: ("good", "Pose check: confirmed"), False: ("bad", "Pose check: not confirmed")}.get(
        verified, ("unsure", "Pose check: uncertain"))
    bits = []
    if pose.get("median_lean_deg") is not None:
        bits.append(f"median lean {_num(pose['median_lean_deg'], 1)} deg")
    if pose.get("max_leg_spread_bh") is not None:
        bits.append(f"max leg spread {_num(pose['max_leg_spread_bh'], 2)} bh")
    if pose.get("usable_frames") is not None:
        bits.append(f"{pose['usable_frames']}/{pose.get('frames_checked', '?')} usable frames")
    detail = f' <span class="why">{_e("; ".join(bits))}</span>' if bits else ""
    note = f'<div class="why">{_e(str(pose["note"]))}</div>' if pose.get("note") else ""
    return f'<div class="lab">Second opinion (pose)</div><div><span class="chip {chip[0]}">{chip[1]}</span>{detail}</div>{note}'


def _metrics_html(ev: dict, cfg: dict) -> str:
    """The measured numbers of an event, with labels in the scenario's units."""
    unit_word = utils.unit_name(cfg)
    rows = []
    for key, val in (ev.get("metrics") or {}).items():
        if key == "peak_time_s":
            rows.append(("Key moment", fmt_time_precise(val)))
        elif key == "zone":
            rows.append(("Zone", str(val)))
        elif key == "fell":
            rows.append(("Went from upright to lying", "Yes" if val else "No (already lying when first seen)"))
        elif key == "contact":
            rows.append(("Possible contact", "YES" if val else "No"))
        elif key == "other_id" and val is not None:
            rows.append(("Other party", utils.entity_name(cfg, val, (ev.get("metrics") or {}).get("other_class"))))
        elif key == "min_ttc_s" and val is None:
            rows.append(("Shortest time to collision", "none (not closing fast enough)"))
        elif key == "min_duration_s" and not val:
            continue                                   # "minimum 0 s" says nothing
        elif key in _METRIC_LABELS:
            label, unit, nd = _METRIC_LABELS[key]
            unit = unit.replace("U", unit_word)
            rows.append((label, f"{_num(val, nd)} {unit}".strip()))
    if not rows:
        return ""
    body = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rows)
    return f'<div class="lab">Measured numbers</div><dl class="metrics">{body}</dl>'


def _captions(ev: dict, cfg: dict) -> tuple[str, str]:
    """(snapshot caption, plot caption) that fit the behaviour."""
    beh = ev.get("behavior")
    word = utils.entity_word(cfg).lower()
    if beh == "near_miss":
        return ("taken at the closest moment: the line joins the two feet and turns red when they are close.",
                "Distance between them (and closing speed) over time, with the near-distance line.")
    if beh == "crowding":
        return ("everyone inside the zone is boxed.", "Count inside the zone over time, with the crowd limit.")
    return (f"thick box = the {word}, line = their path during the event.",
            "Measured value over time, with the rule line(s).")


def _event_card(ev: dict, out_dir: Path, weights: dict | None, cfg: dict, chain: dict | None = None) -> str:
    beh = ev["behavior"]
    color = BEHAVIOR_COLORS_HEX.get(beh, "#2a56c6")
    conf = float(ev.get("confidence") or 0.0)
    start = ev.get("start") or fmt_time(ev["start_s"])
    end = ev.get("end") or fmt_time(ev["end_s"])
    metrics = ev.get("metrics") or {}

    snap = _img(out_dir, ev.get("snapshot"), f"Snapshot of event {ev['event_id']}")
    plot = _img(out_dir, ev.get("speed_plot"), f"Evidence plot of event {ev['event_id']}")
    media = ""
    if snap or plot:
        snap_t = ev.get("snapshot_time_s")
        cap = f"Snapshot at {fmt_time_precise(snap_t)}" if snap_t is not None else "Snapshot"
        snap_cap, plot_cap = _captions(ev, cfg)
        media = '<div class="media">'
        media += f"<figure>{snap}<figcaption>{_e(cap)}: {_e(snap_cap)}</figcaption></figure>" if snap else "<div></div>"
        media += f"<figure>{plot}<figcaption>{_e(plot_cap)}</figcaption></figure>" if plot else ""
        media += "</div>"

    parts_html = ""
    parts = ev.get("confidence_parts") or {}
    if parts:
        parts_html = '<div class="parts">'
        for key, (label, wkey) in _PART_LABELS.items():
            if key in parts:
                w = f" (weight {weights[wkey]:g})" if weights and wkey in weights else ""
                parts_html += (f"<span>{_e(label)}{_e(w)}</span>{_bar(float(parts[key]), '#2a56c6')}"
                               f"<b>{_num(parts[key])}</b>")
        parts_html += "</div>"
    base = ev.get("baseline")
    base_html = ""
    if base and (base.get("note") or base.get("anomaly_score") is not None):
        base_html = (f'<div class="lab">Compared with this scene</div><div class="note">'
                     f'<span class="chip info">anomaly score {_num(base.get("anomaly_score"), 1)}</span>'
                     f'{_e(str(base.get("note") or ""))}</div>')

    # extra chips: zone, fall details, who is in the group
    chips = ""
    if ev.get("zone"):
        chips += f'<span class="chip info">zone: {_e(str(ev["zone"]))}</span>'
    if beh == "fall":
        chips += (f'<span class="chip bad">fell: upright to lying (w/h {_num(metrics.get("upright_aspect_before"), 1)} '
                  f'to {_num(metrics.get("max_aspect"), 1)})</span>' if metrics.get("fell") else
                  f'<span class="chip unsure">lying down (w/h {_num(metrics.get("max_aspect"), 1)}), '
                  f'fall not seen</span>')
    contact = '<span class="badge contact">POSSIBLE CONTACT</span>' if (beh == "near_miss" and metrics.get("contact")) else ""

    group_html = ""
    if ev.get("entity_id") is None and ev.get("entities"):
        ids = list(ev["entities"])
        names = ", ".join(utils.entity_name(cfg, i) for i in ids[:10]) + (f" and {len(ids) - 10} more" if len(ids) > 10 else "")
        group_html = f'<div class="lab">Who was in the group</div><div class="note">{_e(names)}</div>'

    partof = ""
    if chain:
        partof = (f'<p class="partof">Part of <a href="#incident-{chain["chain_id"]}">incident {chain["chain_id"]}: '
                  f'{_e(str(chain.get("title") or ""))}</a></p>')

    return f"""
<article class="card" id="event-{ev['event_id']}" style="--c:{color};--bc:{_badge_text_color(color)}">
  <div class="head">
    <span class="badge">{_e(_behavior_label(ev, cfg))}</span>{_sev_badge(ev.get('severity'))}{contact}
    <h3>Event {ev['event_id']} &middot; {_e(_who(ev, cfg))}</h3>{chips}
    <span class="when">{_e(start)} &ndash; {_e(end)} &middot; {_num(ev.get('duration_s'), 1)} s</span>
  </div>
  {partof}
  {media}
  <div class="cols">
    <div>
      <p class="evidence">{_e(str(ev.get('evidence') or ''))}</p>
      {_metrics_html(ev, cfg)}
    </div>
    <div>
      <div class="lab">Confidence</div>
      <div class="conf"><span class="n">{conf:.2f}</span>{_bar(conf, _conf_color(conf))}</div>
      {parts_html}
      {group_html}
      {base_html}
      {_pose_html(ev)}
    </div>
  </div>
</article>"""


def _incident_card(chain: dict, events_by_id: dict, cfg: dict) -> str:
    """One incident (chain): title, severity, story and links to its event cards."""
    sev = chain.get("severity")
    color = SEVERITY_COLORS.get(sev, "#888888")
    steps = []
    for n, eid in enumerate(chain.get("event_ids", [])):
        ev = events_by_id.get(eid)
        beh_color = BEHAVIOR_COLORS_HEX.get((ev or {}).get("behavior"), "#888888")
        label = f"{_behavior_text(ev, cfg)} {ev.get('start') or ''}-{ev.get('end') or ''}" if ev else ""
        steps.append(f'<a class="step" style="--c:{beh_color}" href="#event-{eid}"><b>Event {_e(str(eid))}</b> {_e(label)}</a>')
    arrow = ' <span aria-hidden="true">&rarr;</span> '
    who = chain.get("entity_name") or utils.entity_name(cfg, chain.get("entity_id"))
    return f"""
<article class="inc" id="incident-{chain['chain_id']}" style="--c:{color};--bc:{_badge_text_color(color)}">
  <div class="head">
    {_sev_badge(sev)}
    <h3>Incident {chain['chain_id']} &middot; {_e(str(chain.get('title') or ''))}</h3>
    <span class="when">{_e(str(chain.get('start') or ''))} &ndash; {_e(str(chain.get('end') or ''))} &middot; {_e(who)}</span>
  </div>
  <p class="story">{_e(str(chain.get('story') or ''))}</p>
  <div class="steps"><span>Evidence:</span> {arrow.join(steps)}</div>
</article>"""


def _table(headers: list[str], rows: list[list[str]], num_cols: tuple[int, ...] = ()) -> str:
    """HTML table from already-escaped cell strings."""
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if j in num_cols else f"<td>{c}</td>"
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _almost_flagged_table(almost: list[dict], cfg: dict) -> str:
    """The 'below the rule' cases (final['near_misses']): behaviour that reached most of a threshold."""
    if not almost:
        return '<div class="empty">None: no behaviour came close to a rule without crossing it.</div>'
    rows = []
    for nm in almost:
        unit = f" {nm.get('unit', '')}".rstrip()
        rows.append([_e(utils.entity_name(cfg, nm.get("track_id"))), _e(_behavior_text(nm.get("behavior"), cfg)),
                     _e(f"{_num(nm.get('value'), 1)}{unit} of {_num(nm.get('threshold'), 1)}{unit}"),
                     _e(str(nm.get("note") or ""))])
    return _table(["Who", "Behaviour", "Reached / rule", "Why it was not flagged"], rows, num_cols=(2,))


def _unusual_table(unusual: list[dict], events: list[dict], cfg: dict) -> str:
    if not unusual:
        return '<div class="empty">Nothing stood out from the scene\'s typical behaviour.</div>'
    rows = []
    for u in unusual:
        mine = [e for e in events if _involves(e, u.get("entity_id"))]
        links = ", ".join(f'<a href="#event-{e["event_id"]}">E{e["event_id"]}</a>' for e in mine)
        status = f"Flagged ({links})" if mine else "Unusual but below every rule: not flagged"
        rows.append([_e(utils.entity_name(cfg, u.get("entity_id"))), _num(u.get("anomaly_score"), 1), status,
                     _e(str(u.get("note") or ""))])
    return _table(["Who", "Anomaly score", "Outcome", "What is unusual"], rows, num_cols=(1,))


def _baseline_line(baseline: dict, cfg: dict) -> str:
    if not baseline or not baseline.get("enabled", True):
        return "Scene baseline is switched off."
    if not baseline.get("active"):
        need = cfg.get("behaviors", {}).get("baseline", {}).get("min_tracks", "?")
        return (f"Scene baseline not active: only {_entity_plural(cfg, int(baseline.get('n_tracks', 0) or 0))} were seen "
                f"(at least {need} are needed to judge what is normal).")
    return (f"Baseline from {_entity_plural(cfg, int(baseline.get('n_tracks') or 0))}: typical speed "
            f"{_num(baseline.get('median_speed_bh_s'), 2)} {utils.unit_name(cfg)}/s, "
            f"typical standing time {_num(baseline.get('median_dwell_s'), 1)} s.")


def _settings_rows(cfg: dict, meta: dict) -> list[list[str]]:
    b = cfg.get("behaviors", {})
    lo, zi, ru, bl = (b.get("loitering", {}), b.get("zone_intrusion", {}), b.get("running", {}), b.get("baseline", {}))
    fa, cr, nm = b.get("fall", {}), b.get("crowding", {}), b.get("near_miss", {})
    ev, vid, ft = cfg.get("events", {}), cfg.get("video", {}), cfg.get("features", {})
    w = ev.get("confidence_weights", {})
    u = utils.unit_name(cfg)
    name = lambda key: utils.behavior_name(cfg, key)          # noqa: E731  (short alias for readability)
    rows = [
        [name("loitering"), f"within {lo.get('radius_bh')} {u} of one spot for at least {lo.get('min_duration_s')} s"],
        [name("zone_intrusion"), f"feet inside a zone polygon for at least {zi.get('min_duration_s')} s"],
        [name("running"), f"starts above {ru.get('start_speed_bh_s')} {u}/s, ends below {ru.get('end_speed_bh_s')} {u}/s, "
                          f"lasting at least {ru.get('min_duration_s')} s"],
    ]
    if fa:
        rows.append([name("fall"), f"box width / height >= {fa.get('down_min_aspect')} for at least "
                                   f"{fa.get('min_down_s')} s (standing means <= {fa.get('upright_max_aspect')} "
                                   f"within {fa.get('transition_s')} s before)"])
    if cr:
        where = "the whole frame" if cr.get("whole_frame") else "a zone"
        rows.append([name("crowding"), f"{cr.get('min_count')} or more inside {where} for at least "
                                       f"{cr.get('min_duration_s')} s"])
    if nm:
        rows.append([name("near_miss"), f"closer than {nm.get('near_distance_bh')} {u} while moving "
                                        f">= {nm.get('min_rel_speed_bh_s')} {u}/s relative to each other, or "
                                        f"time-to-collision under {nm.get('ttc_s')} s; possible contact under "
                                        f"{nm.get('contact_bh')} {u}"])
    rows += [
        ["Scene baseline", f"unusual if robust z-score >= {bl.get('z_threshold')} "
                           f"(needs {bl.get('min_tracks')}+ {_entity_plural_word(cfg)})"],
        ["Incident chains", f"one entity's events linked when the next starts within "
                            f"{(cfg.get('chains') or {}).get('max_gap_s')} s of the previous one ending"],
        ["Event merging", f"pieces of one behaviour closer than {ev.get('merge_gap_s')} s are joined; "
                          f"'almost flagged' shown from {ev.get('near_miss_ratio')} of a threshold"],
        ["Confidence", f"{w.get('margin')} x margin + {w.get('duration')} x duration + {w.get('detection')} x detection"],
        ["Smoothing", f"{ft.get('smoothing_s')} s foot-point average; speed over {ft.get('speed_window_s')} s"],
        ["Video", f"resized to {vid.get('resize_width')} px wide, every {vid.get('frame_stride')} frame(s)"],
    ]
    model = meta.get("model") or cfg.get("model", {}).get("weights")
    if model:
        rows.append(["Detector", f"{model}" + (f" on {meta['device']}" if meta.get("device") else "")])
    zones = meta.get("zones")
    if zones:
        names = [z.get("name", "zone") if isinstance(z, dict) else str(z) for z in zones]
        rows.append(["Restricted zones", ", ".join(names)])
    if meta.get("privacy"):
        rows.append(["Privacy", "faces blurred in the video, highlight reel, snapshots and heatmap"])
    return [[_e(a), _e(str(c))] for a, c in rows]


def _people_table(tracks_summary: list[dict], events: list[dict], cfg: dict) -> str:
    if not tracks_summary:
        return f'<div class="empty">No {_e(_entity_plural_word(cfg))} were tracked.</div>'
    unit = utils.unit_name(cfg)
    rows = []
    for t in sorted(tracks_summary, key=lambda d: (d.get("first_seen") or 0, d.get("id") or 0))[:40]:
        n_ev = sum(1 for e in events if _involves(e, t.get("id")))
        rows.append([_e(utils.entity_name(cfg, t.get("id"), t.get("cls"))),
                     _e(f"{fmt_time(t.get('first_seen'))} - {fmt_time(t.get('last_seen'))}"),
                     _num(t.get("duration_s"), 1), _num(t.get("median_speed_bh_s"), 2),
                     _num(t.get("p90_speed_bh_s"), 2), _num(t.get("max_dwell_s"), 1), str(n_ev)])
    more = (f'<p class="lead">Showing 40 of {len(tracks_summary)}.</p>' if len(tracks_summary) > 40 else "")
    return _table(["Who", "Visible", "Seconds", f"Median speed ({unit}/s)", f"Fast speed, 90th pct ({unit}/s)",
                   "Longest stand (s)", "Events"], rows, num_cols=(1, 2, 3, 4, 5, 6)) + more


def _build_html(final: dict, meta: dict, tracks_summary: list[dict], out_dir: Path, summary: str, cfg: dict) -> str:
    events = final.get("events", []) or []
    incidents = final.get("incidents", []) or []
    events_by_id = {e["event_id"]: e for e in events}
    chain_of = _chain_lookup(incidents)
    n_people = meta.get("n_people", len(tracks_summary))
    dur = meta.get("duration_s")
    n_pose_ok = sum(1 for e in events if e.get("verified") is True)
    pose_ran = bool(meta.get("pose")) or any(e.get("verified") is not None for e in events)
    n_loose = len([e for e in events if e.get("event_id") not in chain_of])
    n_incidents = len(incidents) + n_loose
    n_near = len([e for e in events if e.get("behavior") == "near_miss"])
    n_contact = len([e for e in events if e.get("behavior") == "near_miss" and (e.get("metrics") or {}).get("contact")])

    tiles = [("Video", f'<div class="v small">{_e(_video_name(meta))}</div>', ""),
             ("Duration", f'<div class="v">{_e(fmt_time(float(dur))) if dur else "n/a"}</div>', ""),
             (f"{_entity_plural_word(cfg).capitalize()} seen", f'<div class="v">{int(n_people)}</div>', ""),
             ("Incidents", f'<div class="v">{n_incidents}</div>', "alert" if n_incidents else "ok"),
             ("Events", f'<div class="v">{len(events)}</div>', "")]
    if n_near:
        extra = f" ({n_contact} possible contact)" if n_contact else ""
        tiles.append((utils.behavior_name(cfg, "near_miss"),
                      f'<div class="v">{n_near}<span class="why">{_e(extra)}</span></div>', "alert"))
    if pose_ran and events:
        tiles.append(("Pose-confirmed", f'<div class="v">{n_pose_ok} / {len(events)}</div>', ""))
    tiles_html = "".join(f'<div class="tile {c}"><div class="k">{_e(k)}</div>{v}</div>' for k, v, c in tiles)

    sum_lines = summary.split("\n")
    sum_html = f'<p class="first">{_e(sum_lines[0])}</p>'
    bullets = [ln[2:] for ln in sum_lines[1:] if ln.startswith("- ")]
    rest = [ln for ln in sum_lines[1:] if not ln.startswith("- ")]
    if bullets:
        sum_html += "<ul>" + "".join(f"<li>{_e(b)}</li>" for b in bullets) + "</ul>"
    sum_html += "".join(f"<p>{_e(r)}</p>" for r in rest)

    weights = (cfg.get("events", {}) or {}).get("confidence_weights")
    cards = "".join(_event_card(e, out_dir, weights, cfg, chain_of.get(e.get("event_id"))) for e in events) if events else (
        '<div class="empty">No events were detected in this video.</div>')

    if incidents:
        incident_cards = "".join(_incident_card(c, events_by_id, cfg) for c in incidents)
    elif events:
        gap = (cfg.get("chains") or {}).get("max_gap_s", 15)
        incident_cards = (f'<div class="empty">No linked incidents: nobody did two things within {_e(str(gap))} s of each '
                          'other. Each event stands alone in the cards below.</div>')
    else:
        incident_cards = '<div class="empty">No incidents.</div>'

    timeline = _img(out_dir, "timeline.png", "Timeline of tracked entities and events")
    heat = _img(out_dir, "heatmap.jpg", "Foot-point heatmap")
    overview = ""
    if timeline:
        overview += f'<div class="full">{timeline}</div>'
    if heat:
        overview += (f'<h2 style="margin-top:20px">Where {_e(_entity_plural_word(cfg))} spent time</h2>'
                     f'<p class="lead">Foot-point density over the first frame: brighter means more time spent there. '
                     f'Red outline = restricted zone.</p><div class="full heat">{heat}</div>')

    reel = final.get("highlight_reel")
    reel_html = ""
    if reel:
        reel_html = (f'<div class="reel noprint"><div><strong>Highlight reel</strong>'
                     f'<span>Only the incidents, captioned: watch this instead of the full video. '
                     f'Keep {_e(str(reel))} in the same folder as this report.</span></div>'
                     f'<a class="btn" href="{_e(str(reel))}">&#9654; Watch the highlight reel</a></div>')

    video_link = ""
    if (Path(out_dir) / "annotated.mp4").exists():
        video_link = '<p class="noprint"><a href="annotated.mp4">Open the annotated video (annotated.mp4)</a></p>'

    privacy_html = ""
    if meta.get("privacy"):
        privacy_html = (f'<div class="privacy"><b>{_e(PRIVACY_NOTE)}.</b> The video, highlight reel, snapshots and '
                        f'heatmap hide faces; detection and tracking still ran on the original frames.</div>')

    title = _scenario_title(meta, cfg)
    speed = ""
    pfps = meta.get("processing_fps")
    if isinstance(pfps, (int, float)) and pfps == pfps and pfps > 0:
        speed = f"Processed at {pfps:.1f} frames/s" + (f" on {meta['device']}" if meta.get("device") else "") + "."
    sub_bits = [f"{meta['fps']:g} fps" if meta.get("fps") else None,
                f"processed every {meta['stride']} frame(s)" if meta.get("stride") else None,
                f"elapsed {meta['elapsed_s']:.1f} s" if isinstance(meta.get("elapsed_s"), (int, float)) else None]
    sub = " &middot; ".join(_e(s) for s in sub_bits if s)
    u = _e(utils.unit_name(cfg))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>Behaviour report - {_e(_video_name(meta))}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<header class="top">
  <h1>Behaviour report</h1>
  {f'<p class="scen">{_e(title)}</p>' if title else ''}
  <p class="sub">Who did what, when, and why it is unusual.{(' ' + sub) if sub else ''}</p>
  {f'<p class="speed">{_e(speed)}</p>' if speed else ''}
  <div class="tiles">{tiles_html}</div>
</header>
{reel_html}
{privacy_html}
<section>
  <h2>Summary for the lifeguard</h2>
  <div class="summary{'' if events else ' clear'}">{sum_html}</div>
  {video_link}
</section>

<section>
  <h2>Incidents</h2>
  <p class="lead">An incident links one {_e(utils.entity_word(cfg).lower())}&rsquo;s events over time into a story, so a
  slow build-up is read as one thing, not as separate alerts. Most serious first in the summary above; here in time order.</p>
  {incident_cards}
</section>

<section>
  <h2>Events with evidence</h2>
  <p class="lead">Each card shows the picture, the measured numbers, and the rule that was crossed.
  The size unit is {u}: the thing&rsquo;s own size in the image, so near and far objects are judged fairly.</p>
  {cards}
</section>

<section>
  <h2>Timeline</h2>
  {overview or '<div class="empty">Timeline image not available.</div>'}
</section>

<section>
  <h2>Almost flagged</h2>
  <p class="lead">Behaviour that reached at least {_e(str((cfg.get('events', {}) or {}).get('near_miss_ratio', 0.7)))} of a rule's threshold but did not become an event.
  (Not the same as the &ldquo;{_e(utils.behavior_name(cfg, 'near_miss'))}&rdquo; behaviour above, which is two things that almost collided.)</p>
  {_almost_flagged_table(final.get('near_misses', []) or [], cfg)}
</section>

<section>
  <h2>Unusual compared with this scene</h2>
  <p class="lead">{_e(_baseline_line(final.get('baseline') or {}, cfg))}</p>
  {_unusual_table(final.get('unusual_tracks', []) or [], events, cfg)}
</section>

<section>
  <h2>{_e(_entity_plural_word(cfg).capitalize())} tracked</h2>
  {_people_table(tracks_summary, events, cfg)}
</section>

<section>
  <h2>Settings used</h2>
  <p class="lead">All thresholds come from config.yaml (and the scenario preset); nothing is hard-coded.</p>
  {_table(['Setting', 'Value'], _settings_rows(cfg, meta))}
</section>

<footer>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by the AquaWatch pipeline. Detection and tracking:
YOLO + ByteTrack; behaviour rules are transparent thresholds on measured geometry, so every incident can be explained with numbers.</footer>
</div></body></html>
"""


# =========================================================================== public entry point

def write_outputs(final: dict, meta: dict, tracks_summary, out_dir) -> dict:
    """Write events.json, summary.txt and report.html into out_dir.

    `tracks_summary` is a list of features.track_summary dicts (a dict of them is accepted too).
    Returns {"events_json", "summary_txt", "report_html", "summary"} with paths as strings.
    """
    out = ensure_dir(out_dir)
    meta = dict(meta or {})
    if isinstance(tracks_summary, dict):
        tracks_summary = list(tracks_summary.values())
    tracks_summary = list(tracks_summary or [])
    meta.setdefault("n_people", len(tracks_summary))
    cfg = _resolve_cfg(meta)

    final = dict(final or {})
    if "incidents" not in final and final.get("events"):     # older results: link the events now
        try:
            import chains
            final["incidents"] = chains.build_chains(final["events"], cfg)
        except Exception as err:                              # optional step: never stop the report
            print(f"      (could not build incident chains: {err})")
            final["incidents"] = []

    summary = make_summary(final, meta, cfg)

    doc = {"video": meta.get("video"), "meta": _jsonable(meta),
           "events": final.get("events", []), "incidents": final.get("incidents", []) or [],
           "near_misses": final.get("near_misses", []),
           "unusual_tracks": final.get("unusual_tracks", []), "baseline": final.get("baseline", {}),
           "highlight_reel": final.get("highlight_reel"), "tracks": tracks_summary}
    write_json(out / "events.json", doc)
    (out / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    page = _build_html(final, meta, tracks_summary, out, summary, cfg)
    (out / "report.html").write_text(page, encoding="utf-8")
    return {"events_json": str(out / "events.json"), "summary_txt": str(out / "summary.txt"),
            "report_html": str(out / "report.html"), "summary": summary}
