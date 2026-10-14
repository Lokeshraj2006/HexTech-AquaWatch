"""AquaWatch - the dashboard (Streamlit): drowning early warning for swimming-pool CCTV.

    streamlit run app.py        (or start_aquawatch.bat next to the project folder)

Single camera: pick a pool video, a detector and (optionally) pool zones in the sidebar, press Analyse video, and read
the analysis dashboard: a one-line status, headline numbers, the annotated video beside the live swimmer status and the
alert cards (one replay slider drives them together), then the detail tabs (the run's command and log is the last
one). The look (light "glass" panels, like macOS) comes from ui_theme.py and .streamlit/config.toml. It never re-implements the analysis: the button starts
`run.py` as a subprocess with exactly the flags you can also type yourself, and the page shows the files run.py
writes (events.json, pool_dashboard.json, annotated.mp4, report.html, ...). The heavy libraries (torch, ultralytics)
are never imported here, so the page stays fast. Multi-camera site: site_ui.py. On every page the floating
"Ask AquaWatch" button (chat_ui.py) opens the chat about what is on the page.

The last run is saved to outputs/_ui/last_run.json, so reloading the page (a new Streamlit session) shows it again.
Bad input (a broken zones file, a file that is not a video, a strange events.json) is explained on the page, never
a traceback. The plain logic (file discovery, command building, reading results) lives in ui_helpers.py.
"""
from __future__ import annotations

import functools
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent             # the project folder; the code is in aquawatch/
if str(ROOT / "aquawatch") not in sys.path:        # work no matter where `streamlit run` was started from
    sys.path.insert(0, str(ROOT / "aquawatch"))

import streamlit as st  # noqa: E402

import cpu_limits  # noqa: E402
cpu_limits.apply()  # below-normal priority + thread budget, inherited by every run.py started from the dashboard

import ui_helpers as ui  # noqa: E402
import utils  # noqa: E402

PITCH = ("Early warning for lifeguards: it tracks every swimmer, watches how their behaviour changes over time, "
         "and raises an evidence-backed alert saying who, where, when and why, out loud.")
# Labels say what each detector is, not how accurate it is: accuracy on our own CCTV is not measured yet
# (docs/RESULTS.md section O). Per frame, the posture model and Standard take about the same time.
DETECTOR_LABELS = {"pool_hd": "Wide CCTV (detailed, slow)", "pool_trained": "Swimmer posture model (fast)",
                   "pool": "Standard (fast, stock model)"}
DETECTOR_HELP = {
    "pool_hd": "Stock YOLO11s at 960 px, every frame: made for far-away swimmers in wide, crowded CCTV. Slowest.",
    "pool_trained": "Our own model trained on pool footage: also tells horizontal from upright swimmers. Good for "
                    "close and medium shots.",
    "pool": "Stock YOLO11n at 640 px: a quick general check, and the one for drawn (synthetic) test clips."}
MAX_EVENT_CARDS = 30                              # event cards drawn before "Show all" (each one loads 2 pictures)
RESULT_TABS = ["Incidents & evidence", "Timeline & heatmap", "Summary", "Full report", "Downloads", "Run log"]

st.set_page_config(page_title="AquaWatch", page_icon=":material/pool:", layout="wide",
                   initial_sidebar_state="expanded")

import ui_theme  # noqa: E402
ui_theme.apply()                                   # the glass look, on both pages


# --------------------------------------------------------------------------- cached readers (keyed by file mtime)

@st.cache_data(max_entries=16, show_spinner=False)
def cached_results(path: str, mtime: float) -> dict:
    """events.json read and made safe (ui.sanitize_results); `mtime` is only part of the cache key, so a file
    that is rewritten by a new run is read again, and an unchanged one is read once, not on every rerun."""
    return ui.sanitize_results(utils.read_json(path))


@st.cache_data(max_entries=16, show_spinner=False)
def cached_text(path: str, mtime: float) -> str:
    """Read a text file (summary.txt); cached by path and modification time."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


@st.cache_data(max_entries=8, show_spinner=False)
def cached_facts(path: str, mtime: float):
    """Duration, fps and size of a video (None when it cannot be opened); cached by path and mtime."""
    return ui.video_facts(path)


@st.cache_data(max_entries=8, show_spinner=False)
def cached_frame(path: str, mtime: float, max_width: int = 960):
    """First frame of a video (BGR array, scale); cached by path and mtime."""
    return ui.first_frame(path, max_width)


@st.cache_data(ttl=300, show_spinner=False)
def cached_scenario(name: str) -> dict:
    """Title, rules and classes of a preset for the sidebar."""
    return ui.scenario_summary(name)


@st.cache_data(max_entries=16, show_spinner=False)
def cached_browser_video(path: str, modified: float) -> str | None:
    """Browser-playable file of the selected video (converted once when needed); `modified` keys it on the mtime."""
    found = ui.browser_video(path)
    return str(found) if found else None


@st.cache_data(ttl=300, show_spinner=False)
def cached_needs(name: str) -> dict:
    """Classes, stride, resize width, weights and max seconds of a preset (config.yaml + the preset, read once)."""
    return ui.scenario_needs(name)


@st.cache_data(max_entries=64, show_spinner=False)
def cached_detections(video: str, out_dir: str, needs: tuple, stride: int, size: tuple, max_seconds: float,
                      n_frames: int, signature: tuple, preset: str | None = None):
    """Which cached detections.json `--reuse` could use; `signature` changes when any detections file changes."""
    classes, resize_width, weights, weights_sha1 = needs
    return ui.find_cached_detections(video, out_dir, list(classes), stride, size, signature,
                                     resize_width=resize_width, weights=weights, max_seconds=max_seconds,
                                     n_frames=n_frames, weights_sha1=weights_sha1, preset=preset)


@st.cache_data(max_entries=32, show_spinner=False)
def cached_weights_sha1(path: str, modified: float) -> str:
    """SHA-1 of a weights file; `modified` (the file's mtime) is part of the key, so a retrained best.pt is
    hashed again at once instead of after cached_needs' 5-minute TTL."""
    return utils.weights_sha1(path)


def weights_sha1_now(need: dict) -> str:
    """Current SHA-1 of a preset's weights file ("" when it is not on disk)."""
    path = need.get("weights_path", need["weights"])
    found = utils.weights_file(path)
    return cached_weights_sha1(str(path), ui.mtime(found)) if found else ""


def find_cached(preset: str, video: Path, facts: dict, stride: int, max_seconds: float):
    """The cached detections.json that `run.py --reuse` could use for this video and preset (None when none)."""
    need = cached_needs(preset)
    # an explicit "max seconds" wins; otherwise the preset's own value (0 = whole video)
    seconds = float(max_seconds) or need["max_seconds"]
    return cached_detections(str(video), ui.default_out_dir(video, preset),
                             (tuple(need["classes"]), need["resize_width"], need.get("weights_path", need["weights"]),
                              weights_sha1_now(need)),
                             int(stride),
                             (facts["width"], facts["height"]), seconds, int(facts["frames"] or 0),
                             ui.detections_signature(), preset)


def note_missing(name: str, why: str = "") -> None:
    """One quiet line for a file that is not in the results folder (the page never fails on a missing file)."""
    st.caption(f":material/info: `{name}` is not in this folder{(' (' + why + ')') if why else ''}.")


# --------------------------------------------------------------------------- result renderers

def show_chain(inc: dict) -> None:
    """One incident chain: severity badge, who, title, time span and the plain-English story."""
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(str(inc.get("severity") or "-").upper(), color=ui.severity_color(inc.get("severity")))
            st.markdown(f"**{ui.md_escape(inc.get('entity_name') or 'Group')}**: {ui.md_escape(inc.get('title', ''))}"
                        f"  ({inc.get('start', '?')} - {inc.get('end', '?')})")
        st.markdown(ui.md_escape(inc.get("story", "")))


def show_event(e: dict, folder: Path, weights: dict, expanded: bool) -> None:
    """One event as an expander: severity, who, behaviour, time, confidence and its parts, evidence, pictures."""
    sev = str(e.get("severity") or "-")
    who = e.get("entity_name") or f"Entity {e.get('entity_id')}"
    what = e.get("behavior_name") or str(e.get("behavior", "")).replace("_", " ").title()
    conf = float(e.get("confidence") or 0.0)
    label = (f"#{e.get('event_id')} | {sev.upper()} | {ui.md_escape(who)} | {ui.md_escape(what)} | "
             f"{e.get('start', '?')}-{e.get('end', '?')} | confidence {conf:.2f}")
    with st.expander(label, expanded=expanded):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge(sev.upper(), color=ui.severity_color(sev))
            st.badge(f"confidence {conf:.2f}", icon=":material/verified:", color="gray")
            if e.get("zone"):
                st.badge(f"zone: {e['zone']}", icon=":material/crop_free:", color="violet")
        duration = e.get("duration_s")
        st.markdown(f"**Who:** {ui.md_escape(who)}  \n**Behaviour:** {ui.md_escape(what)}  \n"
                    f"**When:** {e.get('start', '?')} to {e.get('end', '?')}"
                    + (f" ({float(duration):.1f} s)" if duration is not None else ""))
        st.markdown("**Why it was flagged**")
        st.markdown(ui.md_escape(e.get("evidence", "")))
        st.caption(ui.confidence_line(e, weights))
        note = (e.get("baseline") or {}).get("note")
        if note:
            st.caption(f"Scene baseline: {note}")
        pose = ui.pose_text(e)
        if pose:
            st.badge(pose[1], icon=":material/accessibility_new:", color=pose[0])
        snapshot = ui.safe_child(folder, e.get("snapshot"))
        plot = ui.safe_child(folder, e.get("speed_plot"))
        if snapshot or plot:
            left, right = st.columns(2)
            if snapshot:
                left.image(snapshot, caption="Evidence snapshot", width="stretch",
                           alt=f"Snapshot of event {e.get('event_id')}: {what}")
            if plot:
                right.image(plot, caption="Measurement over time, with the rule threshold", width="stretch",
                            alt=f"Plot of the measurement behind event {e.get('event_id')}")
        else:
            note_missing(f"snapshots/e{e.get('event_id')}.jpg and its plot")


def show_near_and_unusual(data: dict, cfg: dict) -> None:
    """The 'almost flagged' cases and the unusual entities: what the system saw but did not report. Each box is shown
    only when it has something: the pool rules fill neither, so a pool run shows no empty '(0)' boxes."""
    almost = data.get("near_misses") or []
    if almost:
        with st.expander(f"Almost flagged ({len(almost)})"):
            st.caption("Behaviour that reached 70 percent of a threshold but did not cross it, so a reader can judge "
                       "the thresholds.")
            rows = [{"Who": utils.entity_name(cfg, a.get("track_id")),
                     "Behaviour": utils.behavior_name(cfg, str(a.get("behavior", ""))),
                     "Reached": f"{a.get('value')} {a.get('unit', '')}".strip(),
                     "Rule": f"{a.get('threshold')} {a.get('unit', '')}".strip(),
                     "Why it was not flagged": a.get("note", "")} for a in almost]
            st.dataframe(rows, hide_index=True, alt="Cases that almost reached a rule")
    unusual = data.get("unusual_tracks") or []
    if unusual:
        with st.expander(f"Unusual compared with the scene ({len(unusual)})"):
            rows = [{"Who": utils.entity_name(cfg, u.get("entity_id")), "Anomaly score": u.get("anomaly_score"),
                     "What is unusual": u.get("note", "")} for u in unusual]
            st.dataframe(rows, hide_index=True, alt="Entities unusual compared with the scene")


@st.fragment
def show_results(folder_str: str, source: str) -> None:
    """Metric tiles and the result tabs for one results folder. A fragment: switching a tab does not rerun the page."""
    folder = Path(folder_str)
    events_path = folder / "events.json"
    try:
        data = cached_results(str(events_path), ui.mtime(events_path))
    except Exception as exc:                         # corrupt or half-written file: say so, do not crash
        st.error(f"Cannot read `{ui.rel_path(events_path)}`: {exc}")
        return
    try:
        render_results(folder, data, source)
    except Exception as exc:                         # last safety net: a surprise in one file never shows a traceback
        st.error(f"Could not show the results in `{ui.rel_path(folder)}` ({type(exc).__name__}: {exc}). "
                 "The files are still in the folder.")


def _detector_label(name: str) -> str:
    """Plain name of a pool preset for people ('Wide CCTV (detailed, slow)'), else its title."""
    return DETECTOR_LABELS.get(name) or cached_scenario(name)["title"]


def _max_risk(dash: dict) -> float:
    """Highest risk any swimmer reached (0 when there is no risk data)."""
    return max((float(r) for curve in (dash.get("risk") or {}).values() for _, r in curve), default=0.0)


def _playable(path: Path) -> str | None:
    """A browser-playable copy of a video file (converted once, cached), or None."""
    if not path.is_file():
        return None
    found = cached_browser_video(str(path), ui.mtime(path))
    return found if found and Path(found).is_file() else None


VIEW_PRESET = {"Wide CCTV over a big pool": "pool_hd", "Close-up, phone or small pool": "pool_trained"}


def _view_picked() -> None:
    """The camera-type answer sets the detector (drawn test clips keep the stock detector, see the sidebar)."""
    preset = VIEW_PRESET.get(st.session_state.get("camera_view"))
    names = ui.sidebar_presets(utils.list_scenarios())
    if preset == "pool_trained" and not ui.trained_model_ready(names):
        preset = "pool"                                  # no trained weights here: the stock fast detector
    if preset in names:
        st.session_state["scenario"] = preset
        st.session_state.pop("_auto_stock", None)


def _risk_help(cfg: dict) -> str:
    """Help for 'Highest risk': a peak is not an alert; the thresholds come from the run's own config."""
    p = {**utils.DEFAULT_CONFIG["pool"], **(cfg.get("pool") or {})}
    return (f"The highest risk any swimmer reached, even for a moment. An alert needs the risk to stay high: "
            f"{float(p['warning_risk']):.0%} or more for {float(p['early_warning_fraction']):.0%} of the last "
            f"{float(p['early_warning_window_s']):g} s (Level 1), {float(p['alert_risk']):.0%} or more held for "
            f"{float(p['hold_s']):g} s (Level 2).")


def render_results(folder: Path, data: dict, source: str) -> None:
    """The analysis dashboard of one results folder: headline numbers, the video beside the live swimmer status and
    the alerts (one replay slider drives the video, the status board and the risk chart), then the detail tabs."""
    import chat_ui
    import distress
    import pool_ui
    m = ui.result_metrics(data)
    cfg = (data.get("meta") or {}).get("config") or {}
    weights = ui.confidence_weights(data)
    dash = pool_ui.load_dashboard(folder)
    video_name = Path(str((data.get("meta") or {}).get("video") or folder.name).replace("\\", "/")).name
    chat_ui.set_context("folder", folder, ui.display_name(video_name))

    feed = (data.get("meta") or {}).get("feed_health")
    if feed and feed.get("problems"):
        feed_note = "camera feed problem"
    elif feed and feed.get("ok"):
        feed_note = "camera feed checked"
    else:
        feed_note = "camera feed not checked"
    ui_theme.overline(f"{source} · {_detector_label(m['scenario'])}")
    st.subheader(ui.display_name(video_name), anchor=False)
    st.caption(" · ".join([f"{m['tracked'] if m['tracked'] is not None else '?'} swimmers tracked", m["duration"],
                           feed_note] + (["faces blurred"] if m["privacy"] else [])))
    if feed and feed.get("problems"):            # never let a broken feed look like a quiet pool
        st.error("**Camera feed problem** - the results cover only the usable part of the video:\n"
                 + "\n".join(f"- {p['text']}" for p in feed["problems"]), icon=":material/videocam_off:")
    if feed and feed.get("warnings"):            # G7: camera moving / looking straight down
        st.warning("**Camera view warning** - results less reliable:\n"
                   + "\n".join(f"- {w['text']}" for w in feed["warnings"]), icon=":material/videocam:")
    if dash:                                     # one sentence first: what a lifeguard must know right now
        tone, sentence = pool_ui.headline(dash)
        with st.container(key=f"headline_{tone}"):
            st.markdown(sentence)
    counts = pool_ui.level_counts(dash) if dash else {}
    with ui_theme.glass("stats"):                # the overview numbers, in one quiet row
        cols = st.columns(5 if dash else 2)
        cols[0].metric("Swimmers tracked", m["tracked"] if m["tracked"] is not None else "?")
        if dash:
            first = min((float(a["alert_time"]) for a in dash.get("alerts") or []), default=None)
            cols[1].metric("Not judged", f"{len(distress.never_judged(dash))} of {len(dash.get('people') or [])}",
                           help="People tracked but never scored by the rules (seen too briefly, too small or "
                                "outside the water). No alert does not mean they are fine: check them by eye.")
            cols[2].metric("Highest risk", f"{_max_risk(dash):.0%}", help=_risk_help(cfg))
            cols[3].metric("First alert", utils.fmt_time(first) if first is not None else "none")
            cols[4].metric("Video length", m["duration"])
        else:
            cols[1].metric("Events", m["events"])
            st.metric("Video length", m["duration"])

    if dash and dash.get("people") and float(dash.get("duration_s") or 0) > 0:
        duration = float(dash["duration_s"])
        # row 1: the video and the swimmer status, side by side and the same height (ui_theme: replay_row is a grid;
        # the board scrolls inside, so it never makes the row taller than the video)
        with st.container(key="replay_row"):
            with ui_theme.glass("video"):
                t = st.slider("Replay the pool at (seconds)", 0.0, duration, pool_ui.default_time(dash), step=0.5,
                              key=f"pool_t::{folder}",
                              help="Moves the video, the swimmer status and the risk chart together.")
                annotated = _playable(folder / "annotated.mp4")
                # a privacy run never falls back to the original video: that would show the faces it blurred
                original = None if annotated or m["privacy"] else \
                    _playable(ROOT / str((data.get("meta") or {}).get("video") or ""))
                if annotated or original:
                    st.video(annotated or original, start_time=int(t),
                             alt="Annotated video: every swimmer boxed with an ID" if annotated else "The analysed video")
                    if not annotated:
                        st.caption("Original video (this run skipped the annotated video).")
                elif m["privacy"]:
                    st.caption("No blurred video for this run (it used 'skip videos'); the original is not shown "
                               "because privacy mode was on.")
                else:
                    note_missing("annotated.mp4", "the run used 'skip videos'")
            with ui_theme.glass("board"):
                pool_ui.status_board(dash, t)
        # row 2: the alerts, full width (the alert cards side by side); row 3: the risk chart, full width
        with ui_theme.glass("alerts"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Alerts**", width="stretch")
                st.toggle("Speak alerts", value=True, key="pool_autoplay",
                          help="Plays the alarm and the spoken alert once when new results appear, on this "
                               "device's speaker.")
            level_metrics(counts, per_row=4)
            with st.container(key="alertgrid"):
                pool_ui.alert_cards(folder, data, dash)
        with ui_theme.glass("chart"):
            pool_ui.risk_chart(dash, t, height=230)
    elif dash:                                   # no replay (nobody tracked): the alert counts still show
        with ui_theme.glass("alerts"):
            level_metrics(counts, per_row=4)

    # Heavy tabs (the embedded report) only run while they are the open tab.
    tabs = st.tabs(RESULT_TABS, on_change="rerun", key="result_tab")
    with tabs[0]:                                                    # Incidents & evidence
        events = data.get("events") or []
        if not events:
            st.success("No incidents flagged in this video.")
        shown = events
        if len(events) > MAX_EVENT_CARDS and not st.toggle(
                f"Show all {len(events)} events (the first {MAX_EVENT_CARDS} are shown)", key="all_events",
                help="Every event card loads two pictures; a long video can have hundreds of events."):
            shown = events[:MAX_EVENT_CARDS]
        for i, ev in enumerate(shown):
            show_event(ev, folder, weights, expanded=(i == 0))
        show_near_and_unusual(data, cfg)

    with tabs[1]:                                                    # Timeline & heatmap
        timeline, heatmap = folder / "timeline.png", folder / "heatmap.jpg"
        if timeline.is_file():
            st.image(timeline, caption="Who was visible when (grey), with the alerts (colour)", width="stretch",
                     alt="Timeline with one bar per swimmer and coloured blocks for alerts")
        else:
            note_missing("timeline.png")
        if heatmap.is_file():
            st.image(heatmap, caption="Where the swimmers were", width="stretch",
                     alt="Heatmap of swimmer activity over the first frame")
        else:
            note_missing("heatmap.jpg")
        if dash:
            st.markdown("**What the system noticed about each swimmer, in order**")
            pool_ui.behaviour_timelines(dash)

    with tabs[2]:                                                    # Summary + highlight reel
        summary = folder / "summary.txt"
        if summary.is_file():
            st.code(cached_text(str(summary), ui.mtime(summary)), language="text", wrap_lines=True)
        else:
            note_missing("summary.txt")
        chains = data.get("incidents") or []
        if chains:
            st.markdown("**Incident chains**")
            for inc in chains:
                show_chain(inc)
        reel = ui.safe_child(folder, data.get("highlight_reel") or "highlights.mp4")
        if reel:
            st.markdown("**Highlight reel** (only the incidents, captioned)")
            st.video(_playable(Path(reel)) or reel, alt="Highlight reel: only the incidents, each one captioned")

    if tabs[3].open:                                                 # Full report
        with tabs[3]:
            report = folder / "report.html"
            if report.is_file():
                # report.html is written by this project's own report.py (every value is HTML-escaped there).
                st.iframe(report, height=1000, alt="Full evidence report")
            else:
                note_missing("report.html")

    with tabs[4]:                                                    # Downloads
        st.caption("Files are read when you click, so nothing is loaded into memory before that.")
        with st.container(horizontal=True):
            for name, mime, icon in (("report.html", "text/html", ":material/article:"),
                                     ("events.json", "application/json", ":material/data_object:"),
                                     ("summary.txt", "text/plain", ":material/description:"),
                                     ("annotated.mp4", "video/mp4", ":material/movie:"),
                                     ("highlights.mp4", "video/mp4", ":material/movie:")):
                path = folder / name
                if path.is_file():
                    st.download_button(name, data=path.read_bytes, file_name=name, mime=mime,
                                       icon=icon, on_click="ignore", key=f"dl_{name}")
            st.download_button("Whole folder (.zip)", data=functools.partial(ui.zip_folder_bytes, folder),
                               file_name=f"{folder.name}.zip", mime="application/zip", type="primary",
                               icon=":material/folder_zip:", on_click="ignore", key="dl_zip")

    with tabs[5]:                                                    # Run log (technical: kept out of the way)
        st.caption(f"Results folder: `{ui.rel_path(folder)}`"
                   + (f" · analysis speed {m['processing_fps']} fps (detection and tracking on this computer)"
                      if m["processing_fps"] else ""))
        run_log(folder)


def level_metrics(counts: dict, per_row: int) -> None:
    """The four alert kinds as numbers (2 x 2 in the Alerts card, one row of 4 otherwise)."""
    items = [("Level 1 · check", "aquatic_warning", "Swimmer may need help: early warning."),
             ("Under water? · urgent", "under_water", "Out of sight in the water, then somebody seen again at the very "
                                                      "same spot: they may have been under the water."),
             ("Level 2 · alarm", "aquatic_distress", "High-risk distress pattern held."),
             ("Level 3 · went under", "submersion", "Lost from view while in distress: possible submersion.")]
    with st.container(key="levels"):
        for start in range(0, len(items), per_row):
            for col, (label, kind, why) in zip(st.columns(per_row), items[start:start + per_row]):
                col.metric(label, counts[kind], help=why)


def run_log(folder: Path) -> None:
    """The last analysis started from this page: command, time and full log (the 'Run log' tab)."""
    last = st.session_state.get("last_run")
    if not last:
        st.caption("No analysis was started from this page in this session: these results were opened from a folder.")
        return
    if (ROOT / last["out_dir"]).resolve() != Path(folder).resolve():
        st.caption(f"This is the log of the last analysis (`{last['out_dir']}`), not of the folder shown above.")
    when = f" at {last['when']}" if last.get("when") else ""
    st.caption(f"Finished in {last['seconds']:.1f} s{when}: `{last['out_dir']}`" if last["ok"] else
               f"Failed (exit code {last['code']}){when}.")
    if last["cmd"]:
        st.code(last["cmd"], language="bash", wrap_lines=True)
        st.caption(f"Run from the project folder with the same Python as this app (`{last['interpreter']}`).")
    st.code("\n".join(x[:500] for x in last["lines"]) or "(no output)", language="text", wrap_lines=True)


# --------------------------------------------------------------------------- state

if "last_run" not in st.session_state:             # dict about the latest Run (command, log, out dir, ok)
    st.session_state["last_run"] = ui.load_last_run()   # a browser reload is a NEW session: take it from the disk
    restored = st.session_state["last_run"]
    if restored:                                    # put the sidebar back as it was, so a reload is not a different job
        label = next((lbl for lbl, p in ui.list_videos() if ui.rel_path(p) == restored["video"]), None)
        if label:
            st.session_state["sample"] = label
        names_now = ui.sidebar_presets(utils.list_scenarios())
        if restored.get("scenario") in names_now:     # (stock pool on real footage reopens with the trained model)
            st.session_state["scenario"] = ui.restored_preset(restored["scenario"], restored.get("video"), names_now)
            if st.session_state["scenario"] != restored["scenario"]:     # say so on the page (shown once, below)
                st.session_state["_reopened_as"] = restored["scenario"]
st.session_state.setdefault("_uploads", {})         # uploaded file id -> saved path ("" = rejected), no rewrite per rerun
if st.session_state.pop("_clear_open_results", False):
    st.session_state["open_results"] = None         # a new run's results replace any folder opened earlier


# --------------------------------------------------------------------------- view: one camera or the whole site

import chat_ui  # noqa: E402
chat_ui.warm_up()                                   # load the chat's text model in the background (first answer: no wait)

VIEWS = ["Single camera", "Multi-camera site"]
if "view" not in st.session_state and st.query_params.get("view") == "site":
    st.session_state["view"] = VIEWS[1]             # a reload (e.g. the light / dark switch) stays on the same page
with st.sidebar:
    ui_theme.brand()
    view = st.segmented_control("View", VIEWS, default=None if "view" in st.session_state else VIEWS[0], key="view",
                                format_func={VIEWS[0]: ":material/videocam: Single camera",
                                             VIEWS[1]: ":material/grid_view: Multi-camera"}.get,
                                label_visibility="collapsed", width="stretch") or VIEWS[0]
if view == VIEWS[1]:
    st.query_params["view"] = "site"
elif "view" in st.query_params:
    del st.query_params["view"]
st.session_state[chat_ui.CONTEXT_KEY] = None        # each page says what the chat is about
if view == VIEWS[1]:                                # the multi-camera page reads the site index; nothing below runs
    import site_ui
    site_ui.render()
    chat_ui.render_fab()
    st.stop()


# --------------------------------------------------------------------------- sidebar

video: Path | None = None
facts = None
zone_mode = "None"
zone_path: Path | None = None                       # a ready zones.json (file or upload); the rectangle is written at Run
rect_zones: list[dict] = []
zone_problem = ""
device = "auto"                                     # CPU on this laptop (auto picks a GPU only when there is one)

with st.sidebar:
    ui_theme.section("Video")
    source = st.segmented_control("Video source", ["Sample", "Upload"], default="Sample", key="source",
                                  label_visibility="collapsed") or "Sample"
    if source == "Upload":
        uploaded = st.file_uploader("Upload a pool video", type=[e.lstrip(".") for e in ui.VIDEO_EXTS], key="upload",
                                    help="Saved to samples/uploads/ so it can be picked again later (limit 200 MB).")
        if uploaded is not None:
            saved = st.session_state["_uploads"].get(uploaded.file_id)
            if saved is None or (saved and not Path(saved).exists()):
                try:
                    path = ui.save_upload(uploaded.name, uploaded.getbuffer())
                    if ui.video_facts(path) is None:             # not a video: do not keep it in samples/uploads/
                        ui.discard_upload(path)
                        path = None
                    saved = str(path) if path else ""
                    st.session_state["_uploads"][uploaded.file_id] = saved
                except OSError as exc:
                    saved = None
                    st.error(f"Could not save the upload: {exc}")
            if saved:
                video = Path(saved)
                st.caption(f"Saved as `{ui.rel_path(video)}`")
            elif saved == "":
                st.error(f"`{ui.safe_filename(uploaded.name)}` cannot be opened as a video. "
                         "Try an mp4 (H.264), mov, avi, mkv or webm file.")
        else:
            st.caption("Choose a video file, or switch back to Sample.")
    else:
        choices = dict(ui.list_videos())
        if choices:
            label = st.selectbox("Sample video", list(choices), key="sample", label_visibility="collapsed",
                                 format_func=lambda lbl: ui.display_name(lbl)
                                 + (" (upload)" if str(lbl).startswith("uploads/") else ""))
            video = choices.get(label)
        else:
            st.info("No videos in samples/: upload one.")

    if video is not None:
        facts = cached_facts(str(video), ui.mtime(video))
        if facts is None:
            st.error("This file cannot be opened as a video.")
        else:                                       # the facts of the picked video, in one line
            st.caption(f":material/movie: {utils.fmt_time(facts['duration_s']) if facts['duration_s'] else '?'} · "
                       f"{facts['fps']:.1f} fps · {facts['width']} × {facts['height']}" if facts["fps"] else
                       f":material/movie: {facts['width']} × {facts['height']}")

    # G7: ask what the camera shows instead of guessing. A wide public-pool CCTV view needs the slow, detailed
    # detector (the fast posture model boxed about 2 of 50+ people there, REAL_WORLD_CHECK C); a close or small view
    # is fine with the fast one. The answer sets the detector; the detector box below can still be changed.
    st.segmented_control("What does this camera show?", list(VIEW_PRESET), key="camera_view",
                         on_change=_view_picked, width="stretch",
                         format_func={"Wide CCTV over a big pool": "Wide view",
                                      "Close-up, phone or small pool": "Close-up"}.get,
                         help="Sets the detector. Wide view (CCTV over a big, crowded pool) uses the detailed detector:"
                              " slower, minutes per camera on this laptop, but it sees far more people. Close-up "
                              "(a phone or a small pool) uses the fast one. Not answered: the detector below is used.")

    # The drawn test clips (samples/synthetic*.mp4) only work with the stock detector: the trained swimmer model
    # finds nobody in drawn swimmers. Picking one switches to the stock pool preset; going back to a real video
    # switches back to the trained model (only when the switch was made here).
    if video is not None:
        current = st.session_state.get("scenario")
        if ui.is_synthetic(video) and current in (None, "pool_trained"):
            st.session_state["scenario"] = "pool"
            st.session_state["_auto_stock"] = True
        elif (not ui.is_synthetic(video) and current == "pool" and st.session_state.pop("_auto_stock", False)
              and ui.trained_model_ready(ui.sidebar_presets(utils.list_scenarios()))):
            st.session_state["scenario"] = "pool_trained"
    names = ui.sidebar_presets(utils.list_scenarios())     # AquaWatch: the pool presets only
    infos = {n: cached_scenario(n) for n in names}
    start = ui.default_preset(names)                     # the trained swimmer model when best.pt is here, else stock pool
    # A default index only on first use: once the key is in session state, Streamlit uses that value (and warns if
    # a default is passed as well).
    pick = {} if "scenario" in st.session_state else {"index": names.index(start) if start in names else 0}
    scenario = st.selectbox("Detector", names, key="scenario", format_func=_detector_label,
                            help="  \n".join(f"**{_detector_label(n)}**: {DETECTOR_HELP.get(n, infos[n]['title'])}"
                                             for n in names), **pick) if names else None
    if scenario:
        hint = ui.preset_hint(scenario, video, names)
        if hint:
            st.caption(":material/lightbulb: " + hint)
        was = st.session_state.get("_reopened_as")
        if was and was != scenario:
            st.caption(f":material/info: Reopened with this detector: the last run used **{_detector_label(was)}**. "
                       "The results shown are from that run until you press Analyse video.")
        elif was:
            st.session_state.pop("_reopened_as", None)              # the user picked the old preset again

    ui_theme.section("Pool zones (optional)")
    # A video that comes with its own zones file (<video>_zones.json: wall, water, deep end) uses it automatically,
    # so a person resting at the wall or a lifeguard on the deck is not judged as a swimmer in trouble.
    # A mode switched on automatically for one video is switched off again for a video without its own file: another
    # video's zones would silently decide where the water is (bug hunt 2026-10-11, ui high).
    if video is not None and st.session_state.get("_zones_auto_for") != str(video):
        st.session_state["_zones_auto_for"] = str(video)
        if ui.suggest_zone_file(video, ui.list_zone_files()) is not None:
            st.session_state["zone_mode"] = "Zones file"
            st.session_state["_zones_auto"] = True
        elif st.session_state.pop("_zones_auto", False) and st.session_state.get("zone_mode") == "Zones file":
            st.session_state["zone_mode"] = "None"
    zone_mode = st.segmented_control("Zone type", ["None", "Rectangle", "Zones file"],
                                     default=None if "zone_mode" in st.session_state else "None", key="zone_mode",
                                     label_visibility="collapsed",
                                     help="Mark the water, the deep end or the shallow end: standing still in a "
                                          "shallow zone is never treated as a warning sign.") or "None"
    if zone_mode != "None" and facts is None:
        st.caption("Pick a readable video first.")
    elif zone_mode == "Rectangle":
        width, height = facts["width"], facts["height"]
        vkey = f"{video.name}_{width}x{height}"       # new sliders (and defaults) for every video
        st.caption(f"Corners in original video pixels ({width} x {height}).")
        c1, c2 = st.columns(2)
        x1 = c1.slider("x1 (left)", 0, width, int(width * 0.25), key=f"zx1_{vkey}")
        y1 = c1.slider("y1 (top)", 0, height, int(height * 0.25), key=f"zy1_{vkey}")
        x2 = c2.slider("x2 (right)", 0, width, int(width * 0.75), key=f"zx2_{vkey}")
        y2 = c2.slider("y2 (bottom)", 0, height, int(height * 0.75), key=f"zy2_{vkey}")
        rect_zones = ui.rect_zone(x1, y1, x2, y2)
        if ui.rect_is_empty(rect_zones):
            zone_problem = "The rectangle has no area: move x1 and x2 (or y1 and y2) apart."
            st.error(zone_problem)
    elif zone_mode == "Zones file":
        zone_files = ui.list_zone_files()
        suggested = ui.suggest_zone_file(video, zone_files)
        if zone_files:
            # no file named after this video: nothing is pre-picked (index None), the person must choose one
            picked = st.selectbox("Zones file found in the project", zone_files, key=f"zfile_{video.name if video else ''}",
                                  index=zone_files.index(suggested) if suggested in zone_files else None,
                                  placeholder="Pick the zones file drawn for THIS video", format_func=ui.rel_path)
            zone_path = picked
        zone_upload = st.file_uploader("...or upload a zones.json", type=["json"], key="zone_upload")
        if zone_upload is not None:
            zone_path, err = ui.save_uploaded_zones(video.stem if video else "zones", zone_upload.getvalue())
            if err:
                zone_problem = f"The uploaded zones file is not usable: {err}"
                st.error(zone_problem)
            else:
                st.caption("Using the uploaded zones file.")
        if zone_path is None and not zone_problem:           # "Zones file" is chosen but there is no file to use
            zone_problem = "No zones file chosen: upload a zones.json, draw a rectangle, or set Zone type to None."
            st.warning(zone_problem)
    if video is not None and facts is not None and zone_mode != "None" and not zone_problem:
        zones_preview: list[dict] = rect_zones if zone_mode == "Rectangle" else []
        if zone_mode == "Zones file" and zone_path is not None:
            zones_preview, err = ui.zones_in_video_pixels(zone_path, (facts["width"], facts["height"]))
            if err:
                zone_problem = err
                st.error(err)
        if not zone_problem:                        # the picture only on request (open while drawing a rectangle)
            frame, scale = cached_frame(str(video), ui.mtime(video))
            if frame is not None:
                try:
                    picture = ui.draw_zones(frame, zones_preview, scale)
                except Exception as exc:          # a strange zone must not take the page down: show the plain frame
                    picture, zones_preview = ui.draw_zones(frame, [], scale), []
                    st.warning(f"The zone could not be drawn ({type(exc).__name__}: {exc}). The analysis can still run.")
                with st.expander("Show the zone on the first frame", expanded=zone_mode == "Rectangle"):
                    st.image(picture, width="stretch", caption="The zone (red) on the first frame",
                             alt="First frame of the selected video with the zone outlined in red")
            if ui.zones_outside_frame(zones_preview, (facts["width"], facts["height"])):
                st.warning("The zone lies completely outside the video frame, so nothing can ever enter it. "
                           "Check the zone's image_size or move the rectangle.")

    privacy = st.toggle("Privacy mode (blur faces)", value=False, key="privacy",
                        help="Blurs the top of every person box in all pictures and videos. The analysis is "
                             "unchanged. Off by default.")
    with st.expander("Advanced"):
        config_stride = cached_needs(scenario)["stride"] if scenario else 2      # the detector's own stride
        stride = st.number_input("Frame stride", min_value=1, max_value=30, value=config_stride, step=1,
                                 key=f"stride_{config_stride}",              # a preset with another default gets a fresh box
                                 help="Analyse every Nth frame. Higher is faster but loses fine detail. "
                                      "The default comes from the detector.")
        max_seconds = st.number_input("Max seconds (0 = all)", min_value=0, max_value=36000, value=0, step=5,
                                      key="max_seconds", help="Only analyse the first S seconds of the video.")
        start_time = st.text_input("Video start time (optional)", key="start_time", placeholder="e.g. 14:30",
                                   help="Clock time of the first frame: alerts then also show the clock time.")
        start_time_ok = None
        if start_time.strip():
            try:
                utils.parse_clock(start_time)
                start_time_ok = start_time.strip()
            except ValueError as exc:
                st.warning(f"Start time ignored: {exc}")
        camera_name = st.text_input("Camera name (optional)", key="camera_name", placeholder="e.g. CAM03 Deep end",
                                    max_chars=60, help="Named in every alert and spoken line, so a lifeguard knows "
                                                       "which camera to look at.")
        reuse = st.toggle("Reuse earlier detections if available", value=True, key="reuse",
                          help="Skips the detector when the same video was analysed before with the same settings: "
                               "seconds instead of minutes.")
        skip_video = st.toggle("Skip videos (faster)", key="skip_video",
                               help="Do not write annotated.mp4 and highlights.mp4.")
    pose = False                                      # the pool presets use their own pose signals

    ready = video is not None and facts is not None
    out_dir = ui.default_out_dir(video, scenario) if ready and scenario else None
    can_run = ready and bool(scenario) and not zone_problem
    readiness = ""                                  # one line: how long Analyse will take (also on the preview)
    if ready and scenario:                          # how long Analyse will take, before pressing it
        found = find_cached(scenario, video, facts, stride, max_seconds) if reuse else None
        if found:
            readiness = "Earlier detections found: the analysis takes seconds."
            st.caption(readiness, help=f"Saved detections: `{ui.rel_path(found)}`")
        else:
            est = ui.yolo_seconds(facts, stride, max_seconds)
            readiness = (("No earlier detections" if reuse else "Reuse is off") + ": the detector will run"
                         + (f", about {est / 60:.1f} min on this laptop." if est > 20 else "."))
            st.caption(readiness)
            need = cached_needs(scenario)
            note = ui.weights_missing_note(need.get("weights_path", need["weights"]))
            if note:
                st.caption(":material/warning: " + note)
            others = [n for n in names if n != scenario and                     # each detector with its own stride
                      find_cached(n, video, facts, cached_needs(n)["stride"], max_seconds)] if reuse else []
            if others:                                # the quickest way out of a long wait: another detector has a cache
                st.caption(":material/lightbulb: Earlier detections exist for: "
                           + ", ".join(f"**{_detector_label(n)}**" for n in others[:3]) + ".")
    run_clicked = st.button("Analyse video", type="primary", icon=":material/play_arrow:", width="stretch",
                            disabled=not can_run, key="run_button",
                            help=f"Results are saved to `{out_dir}`." if out_dir else None)
    if ready and zone_problem:
        st.caption("Fix the zone to enable the analysis.")

    ui_theme.section("Previous results")
    result_dirs = ui.sidebar_result_dirs(ui.list_result_dirs())   # AquaWatch: pool results only
    st.selectbox("Open previous results", [ui.rel_path(d) for d in result_dirs], index=None, key="open_results",
                 placeholder="Open a finished analysis", label_visibility="collapsed",
                 help="Shows finished results without running anything.")
    ui_theme.footer()


# --------------------------------------------------------------------------- header, run, preview

st.title("Single camera", anchor=False)

# The picked video plays here until it is analysed (the results below may still be of another video).
_last = st.session_state.get("last_run")
_shown = st.session_state.get("open_results") or (_last["out_dir"] if _last and _last.get("ok") else None)
if ready and not st.session_state.get("open_results") and _shown != out_dir:
    with ui_theme.glass("preview"):
        ui_theme.overline("Selected video · not analysed yet")
        st.subheader(ui.display_name(video.name), anchor=False)
        st.caption(" · ".join(p for p in (
            utils.fmt_time(facts["duration_s"]) if facts["duration_s"] else "",
            f"{facts['fps']:.0f} fps" if facts["fps"] else "", f"{facts['width']} × {facts['height']}",
            readiness, "press Analyse video in the sidebar") if p))
        clip = _playable(video)
        if clip:
            st.video(clip, alt="The selected video, before the analysis")
        else:
            st.caption("This video cannot be played in the browser here; it can still be analysed.")

if run_clicked and can_run:
    st.session_state.pop("_reopened_as", None)          # the "reopened with another preset" note no longer applies
    run_error = None
    started = time.time()
    try:
        if zone_mode == "Rectangle":
            zone_arg = ui.write_zone_file(video.stem, rect_zones, (facts["width"], facts["height"]))
        else:
            zone_arg = zone_path if zone_mode == "Zones file" else None
        found = find_cached(scenario, video, facts, stride, max_seconds) if reuse else None
        notes = [ui.preserve_detections(out_dir, found),             # never lose expensive detections to another preset
                 ui.seed_detections(found, out_dir)]
        notes = [n for n in notes if n]                              # log lines written by the app itself
        cmd = ui.build_command(video, scenario, out_dir, zones=zone_arg, privacy=privacy, device=device,
                               stride=int(stride) if int(stride) != config_stride else None,
                               max_seconds=float(max_seconds), reuse=reuse, no_video=skip_video, pose=pose,
                               start_time=start_time_ok, camera_name=camera_name)
        log = deque(notes, maxlen=12)                                  # only the newest lines are shown live
        with st.status("Analysing the video ...", expanded=True) as status:
            st.caption("To cancel, change any setting or press Stop at the top right: the analysis is stopped.")
            bar = st.progress(0.0, text="Starting")
            tail = st.empty()
            steps = ui.Progress()
            drawn = [0.0, 0]                           # when the log box was last redrawn, and how many lines it shows
            count = [len(log)]                         # lines received so far

            def draw_tail(force: bool = False) -> None:
                """Show the newest log lines; at most 5 times a second, so a chatty run cannot flood the browser."""
                if count[0] != drawn[1] and (force or time.time() - drawn[0] > 0.2):
                    drawn[0], drawn[1] = time.time(), count[0]
                    tail.code("\n".join(x[:300] for x in log), language="text")

            def show_label() -> None:
                """While run.py is silent: status title with the current step and the seconds since Run was pressed."""
                status.update(label=f"Analysing: {steps.label} ({time.time() - started:.0f} s)", expanded=True)
                draw_tail(force=True)                  # also catches lines that arrived inside the 0.2 s window

            def on_line(line: str) -> None:
                """Called for every output line of run.py: move the bar and show the newest lines."""
                log.append(line)
                count[0] += 1
                fraction = steps.feed(line)
                if fraction is not None:
                    bar.progress(fraction, text=steps.label)
                    show_label()
                draw_tail()

            code, lines = ui.run_pipeline(cmd, on_line, on_idle=show_label)
            events_file = ROOT / out_dir / "events.json"
            ok = code == 0 and events_file.is_file() and ui.mtime(events_file) >= started - 2.0
            elapsed = time.time() - started
            if ok:
                bar.progress(1.0, text="Done")
                status.update(label=f"Finished in {elapsed:.1f} s", state="complete", expanded=False)
            else:
                status.update(label=f"The analysis failed (exit code {code})", state="error", expanded=True)
        full_log = notes + lines
        st.session_state["last_run"] = {"ok": ok, "code": code, "out_dir": out_dir, "cmd": ui.display_command(cmd),
                                        "interpreter": cmd[0], "lines": full_log[-400:], "seconds": elapsed,
                                        "video": ui.rel_path(video), "scenario": scenario,
                                        "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        st.session_state["_clear_open_results"] = ok
    except Exception as exc:                          # never crash the app because a run failed
        run_error = f"{type(exc).__name__}: {exc}"
        st.session_state["last_run"] = {"ok": False, "code": -1, "out_dir": out_dir, "cmd": "", "interpreter": "",
                                        "lines": [run_error], "seconds": time.time() - started,
                                        "video": ui.rel_path(video), "scenario": scenario,
                                        "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    ui.save_last_run(st.session_state["last_run"])     # so a page reload (a new session) still shows this run
    st.rerun()                                         # redraw with the new results folder and an updated sidebar list

last = st.session_state.get("last_run")
picked = st.session_state.get("open_results")
showing = None                                         # (folder, source) of the results shown below
if picked:
    showing = (ROOT / picked, "Opened folder")
elif last and last["ok"] and (ROOT / last["out_dir"] / "events.json").is_file():
    showing = (ROOT / last["out_dir"], "Last analysis")

if last and not last["ok"]:                     # a failure stays visible, with its full log one click away
    tail_text = ui.last_lines(last["lines"], 20).replace("```", "'''")
    st.error(f"The last analysis failed (exit code {last['code']}). Last 20 lines:\n```text\n{tail_text}\n```")
    with st.expander("Full log of the failed analysis", icon=":material/terminal:"):
        if last["cmd"]:
            st.code(last["cmd"], language="bash", wrap_lines=True)
        st.code("\n".join(x[:500] for x in last["lines"]) or "(no output)", language="text", wrap_lines=True)


# --------------------------------------------------------------------------- results

if showing:
    if showing[1] == "Last analysis" and out_dir and last["out_dir"] != out_dir:
        st.info(f"These results belong to the last analysis (`{last['out_dir']}`), not to the video and detector "
                f"selected now. Press Analyse video to analyse the current selection.", icon=":material/history:")
    show_results(str(showing[0]), showing[1])
else:
    with ui_theme.glass("empty"):
        st.markdown(f"**No results yet.** {PITCH}")
        st.caption("Results appear here after an analysis, or open previous results from the sidebar.")

chat_ui.render_fab()
