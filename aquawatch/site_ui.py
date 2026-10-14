"""Multi-camera site view of the AquaWatch dashboard (app.py calls render() when "Multi-camera site" is chosen).

It only READS what the pipeline wrote: the site's event index (ingest.py + geometry.py), each camera's run folder
(pool_dashboard.json, events.json, annotated.mp4, snapshots) and the user's saved references (memory.py).
Data is kept apart by kind, with a banner on every page:
  * real footage (incl. the VIRTUAL demo = crops of one real recording),
  * SYNTHETIC (generated video) -> "SYNTHETIC DEMONSTRATION - NOT REAL-WORLD VALIDATION".
The cartoon multicam set is a logic-test fixture (detections come from its script) and is never listed.
Layout (glass redesign 2026-10-13): the sidebar picks the site and lists its cameras, saved places and "Add a site";
the page shows the site header, its numbers and the panels as glass cards (ui_theme.py).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import cameras as camreg
import distress
import identity
import memory
import pool_ui
import ui_helpers as ui
import ui_theme
import utils
from event_index import EventIndex
from ps_conversation import NO_ALIGNMENT, SiteConversation, behaviour_text

ROOT = Path(__file__).resolve().parent.parent   # the project folder
OUTPUTS = ROOT / "outputs"
FIXTURE_SITES = {"multicam"}          # cartoon logic-test fixture: its detections come from the scene script, not CV
BANNERS = {"synthetic": ("red", "SYNTHETIC DEMONSTRATION — NOT REAL-WORLD VALIDATION"),       # (badge colour, words)
           "virtual": ("orange", "VIRTUAL MULTI-CAMERA DEMO — 4 crops of ONE real recording, not 4 physical cameras"),
           "real": ("green", "REAL FOOTAGE"),
           "unlabelled": ("gray", "UNLABELLED DATASET — treat every result as not validated")}
STATUS_WORDS = {"🟢": "Verified", "🟡": "Partly validated", "🔴": "Not available"}   # validation table, in words
STATUS_CHIP = {"answered": ("answered", "green", ":material/check_circle:"),          # (words, colour, icon)
               "no_evidence": ("no evidence", "gray", ":material/search_off:"),
               "needs_clarification": ("needs your answer", "orange", ":material/help:"),
               "unsupported": ("not supported", "red", ":material/block:")}
SOURCES_INLINE = 3                    # cited moments listed under an answer (all of them are in "Sources")


# ------------------------------------------------------------------ plain helpers (no Streamlit)

def site_kind(label: str | None) -> str:
    """synthetic / virtual / real / unlabelled from a dataset label."""
    text = str(label or "").upper()
    for word, kind in (("SYNTHETIC", "synthetic"), ("VIRTUAL", "virtual"), ("REAL", "real")):
        if word in text:
            return kind
    return "unlabelled"


def list_sites(outputs: Path = OUTPUTS) -> list[dict]:
    """Every outputs/<site>/aquawatch.sqlite3 (fixtures excluded) with its name, label and kind, oldest first (a new
    site is added at the end of the list and does not become the default)."""
    sites = []
    for db in sorted(Path(outputs).glob("*/aquawatch.sqlite3"), key=lambda p: (p.stat().st_ctime, str(p))):
        if db.parent.name in FIXTURE_SITES:
            continue
        index = EventIndex(db)
        label = index.get_meta("dataset_label")
        sites.append({"path": str(db), "dir": db.parent, "name": index.get_meta("site") or db.parent.name,
                      "label": label, "kind": site_kind(label)})
    return sites


def max_risk_series(dash: dict, step: float = 0.5) -> list[tuple[float, float]]:
    """Highest risk of any track of one camera on a time grid (from pool_dashboard.json risk curves)."""
    duration = float(dash.get("duration_s") or 0.0)
    grid = [round(i * step, 2) for i in range(int(duration / step) + 1)]
    out = []
    for t in grid:
        best = 0.0
        for curve in (dash.get("risk") or {}).values():
            value = 0.0
            for tt, r in curve:
                if tt > t:
                    break
                value = r
            if curve and curve[-1][0] + step >= t >= curve[0][0]:
                best = max(best, float(value))
        out.append((t, best))
    return out


def clock_span(cams: list[dict]) -> str:
    """'15:00:00 - 15:00:21': first camera start to last camera end (registry times), '' when they are unknown."""
    from datetime import datetime, timedelta
    spans = []
    for c in cams:
        try:
            start = datetime.fromisoformat(str(c.get("start_time")))
        except (TypeError, ValueError):
            continue
        spans.append((start, start + timedelta(seconds=float(c.get("duration_s") or 0))))
    if not spans:
        return ""
    return f"{min(s for s, _ in spans):%H:%M:%S} - {max(e for _, e in spans):%H:%M:%S}"


# ------------------------------------------------------------------ cached readers

@st.cache_data(show_spinner=False, max_entries=32)
def _json(path: str, mtime: float):
    """A JSON file (cached per version)."""
    return utils.read_json(path)


def _read(path: Path):
    """JSON or None when the file is missing."""
    return _json(str(path), path.stat().st_mtime) if path.is_file() else None


@st.cache_data(show_spinner=False, max_entries=16)
def _playable(path: str, mtime: float) -> str | None:
    """Browser-playable copy of a video (H.264), or None."""
    out = ui.browser_video(path)
    return str(out) if out else None


def site_engine_key(path) -> str:
    """Session key of a site's conversation engine: changes when the index file changes (a re-ingest), so a tab kept
    open never answers from an old copy of the camera rows. chat_ui uses the same key."""
    p = Path(path)
    return f"site_engine::{path}::{p.stat().st_mtime if p.is_file() else 0}"


def _engine(site: dict) -> SiteConversation:
    """One conversation engine per site and browser session (keeps a pending clarification)."""
    key = site_engine_key(site["path"])
    if key not in st.session_state:
        st.session_state[key] = SiteConversation(site["path"])
    return st.session_state[key]


def _track_text(c: dict) -> str:
    """'Track #4 (cam02)', or 'whole frame' for an open-vocabulary frame match."""
    return identity.display_track(c["track_id"], c["camera_id"]) if c.get("track_id") is not None else "whole frame"


def _score_text(c: dict) -> str:
    """Risk of an alert as a percentage, or the CLIP match score of an open-vocabulary hit."""
    value = float(c.get("confidence") or 0)
    return f"match {value:.2f}" if str(c.get("event_state") or "").startswith("open-vocabulary") else f"{value:.0%}"


def _show_answer(question: str, result: dict) -> None:
    """One question / answer pair as chat bubbles: the status chip, the answer, the first cited moments in one line
    each (all of them under "Sources"), a short clip of the first one and its snapshots."""
    with st.chat_message("user"):
        st.markdown(ui.md_escape(question))
    with st.chat_message("assistant"):
        words, colour, icon = STATUS_CHIP.get(result["status"], (result["status"], "gray", None))
        st.badge(words, icon=icon, color=colour)
        st.markdown(result["answer"].replace("\n", "  \n"))
        cites = result["citations"]
        if cites:
            st.caption("  \n".join(
                f":material/videocam: **{ui.md_escape(c['camera_name'])}** · {str(c.get('wall_time') or '')[11:22]} · "
                f"{_track_text(c)} · {ui.md_escape(str(c.get('event_state') or ''))} · {_score_text(c)}"
                for c in cites[:SOURCES_INLINE]) + (f"  \n+{len(cites) - SOURCES_INLINE} more under Sources"
                                                    if len(cites) > SOURCES_INLINE else ""))
            with st.expander(f"Sources ({len(cites)})", icon=":material/fact_check:"):
                st.dataframe([{"camera": c["camera_name"], "time": str(c.get("wall_time") or "")[11:22],
                               "video s": round(float(c.get("time_s") or 0), 2),
                               "track": _track_text(c),
                               "event": c.get("event_state"), "risk / match": _score_text(c),
                               "verification": c.get("verification"), "evidence": c.get("evidence"),
                               "snapshot": c.get("snapshot"), "source video": c.get("source")}
                              for c in cites], hide_index=True, width="stretch")
        clip = next((c for c in cites if c.get("clip")), None)
        if clip is not None:                       # the first cited moment as a short looping clip
            path = Path(clip["clip"]["video"])
            playable = _playable(str(path), path.stat().st_mtime) if path.is_file() else None
            if playable:
                st.video(playable, start_time=int(clip["clip"]["start_s"]), end_time=int(clip["clip"]["end_s"]) + 1,
                         loop=True)
                st.caption(f"Clip: {clip['camera_name']}, video {clip['clip']['start_s']:.0f}-"
                           f"{clip['clip']['end_s']:.0f} s around {_track_text(clip)} (annotated video: boxes drawn,"
                           f" faces blurred when privacy mode was on)")
        shots = [c for c in cites if c.get("snapshot") and (ROOT / c["snapshot"]).is_file()][:4]
        if shots:
            cols = st.columns(len(shots))
            for col, c in zip(cols, shots):
                col.image(str(ROOT / c["snapshot"]), caption=f"{_track_text(c)}"
                          f" at {str(c.get('wall_time') or '')[11:22]}")


# ------------------------------------------------------------------ panels

def _cameras_panel(site: dict, cams: list[dict]) -> None:
    """Current video, active swimmers at a moment, the site risk timeline and the camera context."""
    duration = max(float(c.get("duration_s") or 0) for c in cams)
    with ui_theme.glass("site_controls"):
        t = st.slider("Site time (s from the first camera's start)", 0.0, max(duration, 0.5), min(15.0, duration),
                      0.5, key=f"site_t::{site['path']}")
        pick = st.segmented_control("Camera", [c["camera_id"] for c in cams], default=cams[0]["camera_id"],
                                    format_func=lambda cid: next(c["name"] for c in cams if c["camera_id"] == cid),
                                    key=f"site_cam::{site['path']}") or cams[0]["camera_id"]
    cam = next(c for c in cams if c["camera_id"] == pick)
    sync = camreg.sync_info(cam)
    local_t = None if sync["sync_status"] == camreg.SYNC_UNAVAILABLE else max(0.0, t - float(sync["offset_s"]))
    # the video and the swimmer list side by side, the same height (ui_theme: site_row is a grid; the list scrolls
    # inside, so it never leaves a gap under the video)
    with st.container(key="site_row"):
        video_box, swim_box = st.container(key="glass_site_video"), st.container(key="glass_site_swimmers")
        with video_box:
            st.markdown(f"**Current video: {cam['name']}** (annotated by the pipeline)")
            annotated = ROOT / str(cam.get("out_dir") or "") / "annotated.mp4"
            playable = _playable(str(annotated), annotated.stat().st_mtime) if annotated.is_file() else None
            if playable:
                st.video(playable, start_time=int(local_t or 0))
            else:
                st.caption("No annotated video for this camera (run ingest without --no-video).")
        with swim_box:
            with st.container(key="site_swim_head"):
                st.markdown(f"**Active swimmers at {t:.1f} s**")
            swim_scroll = st.container(key="site_swim_scroll")
        with swim_scroll:
            for c in cams:
                dash = _read(ROOT / str(c.get("out_dir") or "") / "pool_dashboard.json")
                info = camreg.sync_info(c)
                if dash is None:
                    st.caption(f"{c['name']}: no dashboard file")
                    continue
                if info["sync_status"] == camreg.SYNC_UNAVAILABLE:
                    st.caption(f"{c['name']}: {NO_ALIGNMENT}")
                    continue
                ct = t - float(info["offset_s"])
                in_view = [p for p in dash.get("people") or [] if p["first_seen"] - 0.5 <= ct <= p["last_seen"] + 0.5]
                alert_ids = {int(a["person_id"]) for a in dash.get("alerts") or []}
                lost = [p for p in dash.get("people") or [] if p not in in_view and p["person_id"] in alert_ids
                        and pool_ui._state_at(ct, p["person_id"], dash)[0] in pool_ui.ALERT_STATES]   # bug 20
                states = [(p, *pool_ui._state_at(ct, p["person_id"], dash)) for p in in_view + lost]
                states.sort(key=lambda x: pool_ui._board_order(x[1], x[2]))         # most urgent first, never cut off
                # the board's own words ("POSSIBLY UNDER WATER", "HIGH RISK · no alert"); a risk number only for states
                # that come from a risk reading, not for an alert
                flagged = [f"{identity.display_track(p['person_id'], c['camera_id'])} {pool_ui.STATE_BADGE[s]}"
                           + ("" if s in pool_ui.ALERT_STATES else f" {r:.0%}")
                           + (" (lost from view)" if p in lost else "")
                           for p, s, r in states if s not in ("NORMAL", "NOT_JUDGED")]
                unjudged = sum(1 for _, s, _ in states if s == "NOT_JUDGED")
                st.markdown(f"**{c['name']}** · {len(in_view)} track(s) in view"
                            + (f" · {unjudged} {pool_ui.NOT_JUDGED_TEXT}" if unjudged else "")
                            + ("  \n" + "  \n".join(flagged[:5]) if flagged else "")
                            + (f"  \n+{len(flagged) - 5} more" if len(flagged) > 5 else ""))
            st.caption("Tracks, not people: one swimmer can be split into several tracks and seen by several cameras. "
                       "Each camera sees only its own part of the pool (see the coverage picture below): a swimmer "
                       "outside it has no box there.")

    try:
        import altair as alt
        import pandas as pd
        rows = []
        for c in cams:
            dash = _read(ROOT / str(c.get("out_dir") or "") / "pool_dashboard.json")
            if dash is None or camreg.sync_info(c)["sync_status"] == camreg.SYNC_UNAVAILABLE:
                continue
            off = float(camreg.sync_info(c)["offset_s"])
            rows += [{"site time (s)": tt + off, "highest risk": r, "camera": c["name"]} for tt, r in max_risk_series(dash)]
        if rows:
            with ui_theme.glass("site_chart"):
                st.markdown("**Risk timeline: highest swimmer risk per camera**")
                lines = alt.Chart(pd.DataFrame(rows)).mark_line(strokeWidth=2).encode(
                    x="site time (s):Q",
                    y=alt.Y("highest risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                    color=alt.Color("camera:N", legend=alt.Legend(orient="bottom")))
                now = alt.Chart(pd.DataFrame({"site time (s)": [t]})).mark_rule(color="#888").encode(
                    x="site time (s):Q")
                st.altair_chart((lines + now).properties(background="transparent").configure_view(strokeWidth=0),
                                height=230)
    except ImportError:
        pass

    with ui_theme.glass("site_context"):
        st.markdown("**Camera context**")
        refs = memory.ReferenceMemory(Path(site["path"]).with_name("references.sqlite3")).all()
        import feed_health                                     # G5: a lost / frozen / dark camera must be visible
        site_feed = feed_health.site_check([{"camera_id": c["camera_id"], "start_time": c.get("start_time"),
                                             "duration_s": c.get("duration_s")} for c in cams], None)
        table, feed_problems, view_warnings = [], [], []
        for c in cams:
            meta = ((_read(ROOT / str(c.get("out_dir") or "") / "events.json") or {}).get("meta") or {})
            info = camreg.sync_info(c)
            feed = meta.get("feed_health") or {}
            problems = (feed.get("problems") or []) + site_feed.get(c["camera_id"], [])
            feed_problems += [f"{c['name']}: {p['text']}" for p in problems]
            view_warnings += [f"{c['name']}: {w['text']}" for w in feed.get("warnings") or []]
            feed_text = ("PROBLEM" if problems else "view warning" if feed.get("warnings") else
                         "ok" if feed.get("ok") else "not checked")
            table.append({"camera": c["name"], "id": c["camera_id"], "starts": str(c.get("start_time"))[11:19],
                          "length (s)": round(float(c.get("duration_s") or 0), 1),
                          "time sync": NO_ALIGNMENT if info["sync_status"] == camreg.SYNC_UNAVAILABLE
                          else f"{info['sync_status']} ({info['offset_s']:+.2f} s)",
                          "feed": feed_text, "tracks": c.get("n_swimmers"), "alerts": c.get("n_alerts"),
                          "model": meta.get("model"),
                          "device": meta.get("device"), "run time (s)": meta.get("runtime_s"),
                          "processing fps": meta.get("processing_fps"),
                          "your places": ", ".join(r["label"] for r in refs if r["camera_id"] == c["camera_id"]) or "-"})
        st.dataframe(table, hide_index=True, width="stretch")
        if feed_problems:
            st.error("**Camera feed problem**:\n" + "\n".join(f"- {p}" for p in feed_problems),
                     icon=":material/videocam_off:")
        if view_warnings:
            st.warning("**Camera view warning** (results less reliable):\n" + "\n".join(f"- {w}" for w in view_warnings),
                       icon=":material/videocam:")
        # the coverage picture written by tools/make_virtual_cameras.py next to the camera videos (when there is one)
        coverage = camreg.resolve_path(str(cams[0].get("video") or "")).parent.parent / "coverage.jpg" if cams else None
        if coverage is not None and coverage.is_file():
            st.image(str(coverage), width="stretch",
                     caption="What each camera sees: its outline on the full recording. A swimmer outside a camera's "
                             "outline has no box on that camera.")


def _events_panel(site: dict, engine: SiteConversation) -> None:
    """All indexed alerts across cameras, and the evidence of one."""
    events = engine.index.query()
    if not events:
        dashes = [d for d in (_read(ROOT / str(c.get("out_dir") or "") / "pool_dashboard.json")
                              for c in engine.cameras.values()) if d]
        unjudged = sum(len(distress.never_judged(d)) for d in dashes)
        total = sum(len(d.get("people") or []) for d in dashes)
        if unjudged:                             # not a green all-clear: some tracks were never looked at
            st.warning(f"No alert was raised on any camera, but {unjudged} of {total} tracks were not judged (seen "
                       f"too briefly, too small or outside the water): check them by eye.", icon=":material/visibility:")
        else:
            st.success("No alert was raised on any camera.", icon=":material/pool:")
        return
    rows = []
    for e in sorted(events, key=lambda e: e.get("alert_wall") or ""):
        partners = engine._geometry_partners(e["camera_id"], e.get("person_id"))
        rows.append({"time": str(e.get("alert_wall") or "")[11:22], "camera": engine.camera_name(e["camera_id"]),
                     "track": identity.display_track(e.get("person_id"), e["camera_id"]),
                     "alert": behaviour_text(e),
                     "risk": f"{float(e.get('confidence') or 0):.0%}",
                     "behaviour (s)": f"{float(e['start_s']):.1f}-{float(e['end_s']):.1f}",
                     "same swimmer on (geometry)": ", ".join(identity.display_track(p["track"], p["camera"])
                                                             for p in partners) or "-"})
    st.dataframe(rows, hide_index=True, width="stretch")
    pick = st.selectbox("Evidence for", range(len(rows)), key=f"site_ev::{site['path']}",
                        format_func=lambda i: f"{rows[i]['time']} {rows[i]['track']} - {rows[i]['alert']}")
    e = sorted(events, key=lambda e: e.get("alert_wall") or "")[pick]
    c1, c2 = st.columns(2)
    snap = ROOT / str(e.get("snapshot") or "")
    if e.get("snapshot") and snap.is_file():
        c1.image(str(snap), caption="Snapshot at the alert")
    plot = snap.parent.parent / "plots" / f"{snap.stem}.png" if e.get("snapshot") else None   # run.py: e2.jpg -> e2.png
    if plot is not None and plot.is_file():
        c2.image(str(plot), caption="Risk over time")
    st.markdown(f"**Evidence:** {e.get('evidence') or '(none recorded)'}")


def _continuity_panel(site: dict, engine: SiteConversation) -> None:
    """Cross-camera continuity: geometry links (certain here), appearance candidates (always unverified)."""
    groups = engine.index.get_meta("geometry_global_ids") or {}
    links = engine.index.links()
    in_group = {(c, int(t)) for members in groups.values() for c, t in members}
    verified = [l for l in links if l["status"] in identity.VERIFIED_STATUSES
                and (l["camera_a"], int(l["track_a"])) in in_group and (l["camera_b"], int(l["track_b"])) in in_group]
    st.markdown(f"**{len(verified)} verified link(s)** between overlapping views (method: known camera geometry), "
                f"**{len(groups)} global id(s)**"
                + (f"; {len(links) - len(verified)} more link(s) contradict each other and are NOT verified"
                   if len(links) > len(verified) else "") + ". A track without a link keeps no global id.")
    if groups:
        st.dataframe([{"global id": gid, "tracks": ", ".join(identity.display_track(t, c) for c, t in members)}
                      for gid, members in groups.items()], hide_index=True, width="stretch")
    same = engine.index.get_meta("geometry_same_swimmer_events") or []
    if same:
        st.markdown("**Alerts that are the same swimmer on several cameras**")
        for g in same:
            st.markdown("- " + "; ".join(f"{x['display']} {float(x['start_s']):.1f}-{float(x['end_s']):.1f} s"
                                         for x in g["events"]) + f"  \n  _{g['evidence'][0]}_")
    agreement = _read(Path(site["dir"]) / "reid_vs_geometry.json")
    if agreement and agreement.get("overall"):
        o = agreement["overall"]
        st.markdown("**Appearance Re-ID (colour histogram): agreement with the geometry links, NOT accuracy**")
        with st.container(horizontal=True):
            st.metric("Partner is top-1 candidate", f"{o.get('top1_agreement', 0):.0%}", border=True)
            st.metric("Partner in top-3", f"{o.get('top3_agreement', 0):.0%}", border=True)
            st.metric("Tracks compared", o.get("n_queries", 0), border=True)
        st.caption("Every appearance link is shown as 'Candidate match - unverified'. Colour alone cannot confirm an "
                   "identity, and these views share lighting and viewpoint, which makes this easier than a real site.")
    with st.expander("Caveats"):
        for line in engine.index.get_meta("geometry_caveats") or []:
            st.markdown(f"- {line}")


def _references_panel(site: dict, engine: SiteConversation) -> None:
    """User-defined places (persistent, versioned, never presented as detected)."""
    st.caption(memory.USER_NOTE)
    with st.form(key=f"ref::{site['path']}", clear_on_submit=True, border=False):
        text = st.text_input("Define a place", placeholder="CAM03 is the deep end")
        if st.form_submit_button("Save") and text.strip():
            result = engine.answer(text)
            (st.success if result["status"] == "answered" else st.warning)(result["answer"].split("\n")[0])
    refs = engine.memory.all()
    if not refs:
        st.info("No places saved yet. Ask about one (e.g. the deep end) and you will be asked once.")
        return
    st.dataframe([{"place": r["label"], "camera": str(r["camera_id"]).upper(), "region": r.get("region") or "-",
                   "version": r["version"], "created": r["created_at"], "updated": r["updated_at"],
                   "source": r["source"], "status": r["verification_status"], "provenance": r["provenance"]}
                  for r in refs], hide_index=True, width="stretch")
    gone = st.selectbox("Forget a place", [None] + [r["label"] for r in refs], key=f"forget::{site['path']}")
    if gone and st.button(f"Forget '{gone}'", key=f"forget_btn::{site['path']}"):
        engine.memory.forget(gone, provenance="dashboard button")
        st.rerun()


def _validation_panel(site: dict, engine: SiteConversation, cams: list[dict]) -> None:
    """What is validated, what only runs, and what is missing (honest status)."""
    metas = [((_read(ROOT / str(c.get("out_dir") or "") / "events.json") or {}).get("meta") or {}) for c in cams]
    n_tracks = sum(int(c.get("n_swimmers") or 0) for c in cams)
    n_alerts = sum(int(c.get("n_alerts") or 0) for c in cams)
    runtime = sum(float(m.get("runtime_s") or 0) for m in metas)
    agreement = (_read(Path(site["dir"]) / "reid_vs_geometry.json") or {}).get("overall") or {}
    kind = site["kind"]
    rows = [
        ("🟢", "Per-camera pipeline (YOLO + ByteTrack + behaviour + alerts) on this footage",
         f"{len(cams)} camera(s), {n_tracks} tracks, {n_alerts} alert(s), {runtime:.0f} s total run time"),
        ("🟡", "Detection / alert accuracy", "Not measured: no human-annotated ground truth for this footage"),
        ("🟢", "Multi-camera index + wall-clock queries", "SQLite index; tests pass (test_event_index, test_ingest)"),
        ("🟢" if kind == "virtual" else "🔴", "Cross-camera links by known geometry",
         f"{len(engine.index.links())} links; certain only because the crop geometry is known exactly"
         if kind == "virtual" else "Needs a measured calibration between overlapping cameras"),
        ("🟡", "Appearance Re-ID (colour histogram)",
         f"Candidates only, always unverified; agreement with geometry top-1 {agreement.get('top1_agreement', 0):.0%},"
         f" top-3 {agreement.get('top3_agreement', 0):.0%}" if agreement else "Candidates only, always unverified"),
        ("🟢", "Ask the Video (grounded answers, clarify once, capability refusal)", "Covered by the local test suite"),
        ("🟢", "Persistent user references (versioned, survive restart)", "references.sqlite3 next to the index"),
        _openvocab_row(Path(site["dir"])),
        ("🔴", "Real multi-camera site (physical cameras, measured sync)", "Impossible now: no such footage"),
    ]
    if kind == "synthetic":
        rows.insert(0, ("🟡", "SYNTHETIC footage", "Shows the system working, not real-world performance"))
    st.dataframe([{"status": STATUS_WORDS.get(a, a), "feature": b, "evidence": c} for a, b, c in rows],
                  hide_index=True, width="stretch")
    st.caption("Verified: implemented and verified on this input · Partly validated: implemented, partially "
               "validated · Not available: not implemented or impossible with the current footage.")
    st.markdown("**Privacy**: everything runs on this computer. No face recognition, no names: swimmers are local "
                "tracker IDs, and a cross-camera link is only made from box geometry (or offered as an unverified "
                "colour candidate). Snapshots and the index stay in `outputs/`.")


def _openvocab_row(site_dir: Path) -> tuple:
    """Validation row of the open-vocabulary search, with the final R1 numbers when the evaluation was run locally."""
    if not (site_dir / "openvocab_index.npz").is_file():
        return ("🔴", "Open-vocabulary search (CLIP)", "No index for this site: run openvocab.py build --db ...")
    runs = [ev for ev in (_read(p) for p in sorted(site_dir.glob("openvocab_eval_r1_set*.json"))) if ev]
    if not runs:
        return ("🟡", "Open-vocabulary search (CLIP, zero-shot)", "Index built; not evaluated on this site")

    def pooled(method: str, key: str) -> str:
        """'12/19': hits of one method summed over the query sets."""
        hits = sum(round((ev["methods"].get(method) or {}).get(key, 0) * ev["n_positive"]) for ev in runs)
        return f"{hits}/{sum(ev['n_positive'] for ev in runs)}"

    def rejected(method: str) -> str:
        """'9/9': "not there" queries the method rejected, over the query sets."""
        hits = sum(round((ev["methods"].get(method) or {}).get("negatives_rejected", 0) * ev["n_negative"])
                   for ev in runs)
        return f"{hits}/{sum(ev['n_negative'] for ev in runs)}"

    return ("🟡", "Open-vocabulary search (OpenAI CLIP ViT-B/32, INT8 embeddings) vs YOLOE / YOLO-World detection",
            f"{sum(ev['n_positive'] for ev in runs)} held-out queries (AI-drafted labels, one clip). CLIP frame ranking "
            f"(used): hit@1 {pooled('baseline_frame', 'hit@1')}, hit@5 {pooled('baseline_frame', 'hit@5')}, ~0.2 s, "
            f"cannot say 'not there'. YOLOE detection: hit@1 {pooled('ground_yoloe', 'hit@1')}, 'not there' rejected "
            f"{rejected('ground_yoloe')}, ~9 s. YOLO-World: hit@1 {pooled('ground_yoloworld', 'hit@1')}. "
            f"Measured locally")


# ------------------------------------------------------------------ page

NEW_SITE_PRESETS = {"Wide CCTV (detailed, slow)": "pool_hd", "Swimmer posture model (fast)": "pool_trained"}


def _new_site_form() -> None:
    """Make a new site from a folder of videos: one camera per video, typed start times, then ingest.py runs here
    (one camera after the other) and the site appears in the list. Nothing is uploaded; the videos stay where they are."""
    st.caption("Each video becomes one camera. The analysis runs on this laptop, one camera after the other "
               "(Wide CCTV: about 30 s of work per second of video). Clock times are what you type: they are "
               "marked 'user declared', not checked.")
    name = st.text_input("Site name", key="ns_name", placeholder="e.g. Main pool, Saturday")
    folder = st.text_input("Folder with the videos", key="ns_folder",
                           placeholder="e.g. samples/main_pool   or   C:\\Videos\\pool")
    videos = ui.site_folder_videos(folder) if folder.strip() else []
    if folder.strip() and not videos:
        st.warning("No videos (.mp4, .mov, .avi, .mkv, .m4v) directly in that folder.")
    if not videos:
        return
    start = st.text_input("Clock time of the first frame (all videos)", key="ns_start",
                          placeholder="2026-10-12 15:00:00",
                          help="Change it per camera in the table if the recordings did not start together.")
    kind = st.selectbox("What is this footage?", list(ui.SITE_KINDS), key="ns_kind",
                        help="Shown on every page and answer of the site, so real and synthetic data never mix.")
    preset = NEW_SITE_PRESETS[st.segmented_control("Detector", list(NEW_SITE_PRESETS), key="ns_preset",
                                                   default="Wide CCTV (detailed, slow)") or
                              "Wide CCTV (detailed, slow)"]
    rows = st.data_editor([{"use": True, "video": p.name, "camera name": ui.short_camera_name(p),
                            "start time": start.strip()} for p in videos],
                          key=f"ns_table::{folder}::{start}", disabled=["video"], hide_index=True, width="stretch")
    privacy = st.toggle("Blur faces in the outputs", value=False, key="ns_privacy")
    by_name = {p.name: p for p in videos}
    chosen = [{"video": by_name[r["video"]], "name": r["camera name"], "start_time": r["start time"]}
              for r in rows if r.get("use")]
    seconds = [float((ui.video_facts(c["video"]) or {}).get("duration_s") or 0.0) for c in chosen]
    minutes = ui.site_estimate_minutes(seconds, preset)
    problems = ui.site_problems(name, chosen)
    for problem in problems:
        st.caption(f":material/error: {problem}")
    if not st.button(f"Create the site and analyse ({len(chosen)} video(s), about {max(1, round(minutes))} min)",
                     type="primary", disabled=bool(problems), key="ns_go", icon=":material/play_arrow:"):
        return
    try:
        path = ui.write_site(ui.build_site_registry(name, kind, preset, chosen))
    except (OSError, ValueError) as exc:
        st.error(f"Could not write the site: {exc}")
        return
    cmd = ui.site_ingest_command(path, name, privacy)
    with st.status(f"Analysing {len(chosen)} camera(s) ...", expanded=True) as status:
        box, tail = st.empty(), []

        def show(line: str) -> None:
            tail.append(line)
            box.code("\n".join(tail[-12:]))
        code, lines = ui.run_pipeline(cmd, on_line=show)
        if code == 0:
            status.update(label="Site added", state="complete")
        else:
            status.update(label=f"ingest.py stopped (exit {code})", state="error")
            st.code("\n".join(lines[-20:]))
            return
    st.session_state["ns_done"] = f"Site '{name}' added: pick it in the Site list."
    st.rerun()


@st.dialog("Add a site from a folder of videos", width="large", icon=":material/video_library:")
def _new_site_dialog() -> None:
    """The new-site form as a dialog, opened from the sidebar."""
    _new_site_form()


def _sidebar(engine: SiteConversation | None, cams: list[dict]) -> None:
    """The rest of the site page's sidebar: its cameras (alerts, tracks), the saved places and 'Add a site'."""
    if cams:
        ui_theme.section("Cameras")
        for c in cams:
            alerts = int(c.get("n_alerts") or 0)
            st.markdown(f"**{ui.md_escape(str(c['name']))}**  \n"
                        + (f":red-badge[{alerts} alert{'' if alerts == 1 else 's'}]" if alerts else
                           ":green-badge[no alerts]")
                        + f" :gray[{int(c.get('n_swimmers') or 0)} tracks]")
    if engine is not None:
        ui_theme.section("Saved places")
        refs = engine.memory.all()
        if refs:
            st.markdown("  \n".join(f":material/place: **{ui.md_escape(r['label'])}** → {str(r['camera_id']).upper()}"
                                     for r in refs))
        else:
            st.caption("None yet. Ask the chat about a place (e.g. the deep end): it asks you once and remembers.")
    ui_theme.section("Sites")
    if st.button("Add a site from videos", icon=":material/video_library:", width="stretch", key="ns_open",
                 help="Each video of a folder becomes one camera; the analysis runs on this laptop."):
        _new_site_dialog()
    ui_theme.footer()


def render() -> None:
    """The multi-camera site page: the site and its cameras in the sidebar; header, numbers and panels on the page."""
    all_sites = list_sites()
    # only the kinds of site that exist are offered (real and synthetic data are never mixed on one page)
    groups = [g for g in ("Real footage", "Synthetic")
              if any((s["kind"] == "synthetic") == (g == "Synthetic") for s in all_sites)]
    site = None
    with st.sidebar:
        ui_theme.section("Site")
        if groups:
            group = groups[0] if len(groups) == 1 else (
                st.segmented_control("Data", groups, default=groups[0], key="site_group", width="stretch",
                                     help="Real and synthetic data are never mixed on one page.") or groups[0])
            sites = [s for s in all_sites if (s["kind"] == "synthetic") == (group == "Synthetic")]
            site = sites[0] if len(sites) == 1 else sites[st.selectbox(
                "Site", range(len(sites)), format_func=lambda i: sites[i]["name"], key=f"site_pick::{group}",
                label_visibility="collapsed")]
            if len(sites) == 1:
                st.caption(site["name"])
    if st.session_state.get("ns_done"):
        st.success(st.session_state.pop("ns_done"), icon=":material/check_circle:")
    if site is None:
        with st.sidebar:
            _sidebar(None, [])
        with ui_theme.glass("site_empty"):
            ui_theme.overline("Multi-camera site")
            st.title("No site yet", anchor=False)
            st.caption("Add one with **Add a site from videos** in the sidebar, or from the command line:")
            st.code("python aquawatch/ingest.py --cameras data/<name>/cameras.yaml "
                    "--db outputs/<name>/aquawatch.sqlite3 --out-root outputs/<name> --reid\n"
                    "python aquawatch/geometry.py --cameras data/<name>/cameras.yaml --geometry data/<name>/geometry.json "
                    "--runs-root outputs/<name> --out outputs/<name>/geometry_links.json "
                    "--db outputs/<name>/aquawatch.sqlite3", language="bash")
        return
    engine = _engine(site)
    cams = list(engine.cameras.values())
    with st.sidebar:
        _sidebar(engine, cams)
    colour, banner = BANNERS[site["kind"]]
    ui_theme.overline("Multi-camera site")
    st.title(site["name"], anchor=False)
    span = clock_span(cams)
    st.caption(f"{len(cams)} camera(s)" + (f" · {span}" if span else ""))
    # generated or unlabelled data is labelled on every page, so it never passes for real footage. The virtual demo
    # says what it is in its own name ("... virtual multi-camera demo (one real recording)"): no banner (the user's
    # choice, 2026-10-13); the Validation status tab still explains it.
    if site["kind"] in ("synthetic", "unlabelled"):
        st.badge(banner + (f" · {site['label']}" if site["label"] and site["kind"] == "unlabelled" else ""),
                 icon=":material/info:", color=colour)
    if not cams:
        st.warning("This index has no cameras yet. Run ingest.py.")
        return
    import chat_ui                                   # the floating "Ask AquaWatch" chat talks about this site
    chat_ui.set_context("site", site["path"], site["name"])
    with ui_theme.glass("stats"):
        cols = st.columns(5)
        cols[0].metric("Cameras", len(cams))
        cols[1].metric("Tracks", sum(int(c.get("n_swimmers") or 0) for c in cams),
                       help="Tracks, not people: one swimmer can be split into several tracks and seen by several "
                            "cameras.")
        cols[2].metric("Alerts", sum(int(c.get("n_alerts") or 0) for c in cams))
        cols[3].metric("Saved places", len(engine.memory.all()))
        cols[4].metric("Footage", f"{max(float(c.get('duration_s') or 0) for c in cams):.1f} s",
                       help="The longest camera recording.")
    tabs = st.tabs(["Cameras & swimmers", "Events & evidence", "Multi-camera continuity", "Places",
                    "Validation status"])
    with tabs[0]:
        _cameras_panel(site, cams)
    with tabs[1]:
        _events_panel(site, engine)
    with tabs[2]:
        _continuity_panel(site, engine)
    with tabs[3]:
        _references_panel(site, engine)
    with tabs[4]:
        _validation_panel(site, engine, cams)
