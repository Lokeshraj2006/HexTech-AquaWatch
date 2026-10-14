"""Stage 4 of the pool pipeline: the lifeguard dashboard pieces of the Streamlit app (app.py places them).

For a results folder that has pool_dashboard.json:
  * alert_cards: one card per alert (Level 1 check / Level 2 alarm / Level 3 went under): who + where + when + risk +
    evidence checklist, the snapshot and risk plot, and the spoken announcement (alarm + voice), auto-played once in
    the browser so it comes out of the dashboard device's speaker;
  * status_board: green / blue / yellow / red per swimmer in view at a chosen time (replay the scene);
  * risk_chart: the distress risk of every swimmer over time, with the alert level and the replay time;
  * behaviour_timelines: what the system noticed about each swimmer, in order;
  * headline: the one-line status of the whole video ("All clear" / "1 urgent alert ...") shown above the numbers.
render_pool_panel draws all of them in one column (the older layout). Cards and chips get container keys that
ui_theme.py styles (alert_<kind>_<n>, swim_<state>_<id>).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

import distress
import utils

STATE_BADGE = {"NORMAL": ":green-badge[NORMAL]", "WATCH": ":blue-badge[WATCH]", "WARNING": ":orange-badge[WARNING]",
               "DISTRESS": ":red-badge[DISTRESS]", "SUBMERSION": ":red-badge[POSSIBLE SUBMERSION]",
               "NOT_JUDGED": ":gray-badge[NOT JUDGED]", "UNDER_WATER": ":red-badge[POSSIBLY UNDER WATER]",
               "HIGH_RISK": ":orange-badge[HIGH RISK · no alert]"}
STATE_DOT = {"NORMAL": "🟢", "WATCH": "🔵", "WARNING": "🟡", "DISTRESS": "🔴", "SUBMERSION": "🆘", "NOT_JUDGED": "⚪",
             "UNDER_WATER": "🌊", "HIGH_RISK": "🟠"}
ALERT_STATES = ("WARNING", "UNDER_WATER", "DISTRESS", "SUBMERSION")   # states that only an alert can give
# a box and an ID but no risk: the rules need about 2 s of steady view of a swimmer big enough to read
NOT_JUDGED_TEXT = "not judged: seen too briefly or too small"
LEVELS = {  # kinds that must never look alike: check (orange) -> under water? (blue) -> alarm (red) -> went under (violet)
    "aquatic_warning": (":material/warning:", "orange", "SWIMMER MAY NEED HELP", "Level 1 of 3 · check this swimmer now"),
    "under_water": (":material/water:", "blue", "POSSIBLY UNDER WATER",
                    "Urgent · out of sight, then seen again at the same spot: go and check now"),
    "aquatic_distress": (":material/emergency:", "red", "HIGH-RISK AQUATIC DISTRESS",
                         "Level 2 of 3 · alarm: distress pattern held"),
    "submersion": (":material/scuba_diving:", "violet", "POSSIBLE SUBMERSION", "Level 3 of 3 · urgent: went under, go now")}


@st.cache_data(show_spinner=False)
def _load(path: str, mtime: float) -> dict:
    """pool_dashboard.json (cached per file version)."""
    return utils.read_json(path)


def load_dashboard(folder: Path) -> dict | None:
    """The folder's pool_dashboard.json, or None when it has none (not a pool run)."""
    path = Path(folder) / "pool_dashboard.json"
    return _load(str(path), path.stat().st_mtime) if path.is_file() else None


def _keep_s(alert: dict, dash: dict) -> float:
    """How long after its end an alert still shows on the status board: a Rule C alert as long as its spot is marked
    on the video (pool.mark_spot_s), any other 1 s."""
    if distress.alert_kind(alert) != "under_water":
        return 1.0
    return float(dash["thresholds"].get("mark_spot_s", utils.DEFAULT_CONFIG["pool"]["mark_spot_s"]))


def _state_at(t: float, pid: int, dash: dict) -> tuple[str, float]:
    """Swimmer state and risk at video time t, from the risk curve and the alerts. NOT_JUDGED (risk 0) when the rules
    have no reading for this swimmer yet: a box and an ID are not enough, it is not the same as NORMAL. A risk reading
    at warning level WITHOUT an alert is HIGH_RISK: WARNING (the Level 1 badge) only ever comes from a real alert, and
    an active alert keeps its state after the swimmer is lost from view (bug hunt 2026-10-11, bug 20)."""
    curve = dash["risk"].get(str(pid)) or []
    judged = bool(curve) and t >= float(curve[0][0])
    risk = 0.0
    for tt, r in curve:
        if tt > t:
            break
        risk = r
    th = dash["thresholds"]
    state = ("HIGH_RISK" if risk >= th["warning"] else "WATCH" if risk >= th["watch"] else "NORMAL") if judged \
        else "NOT_JUDGED"
    active = False                                   # an alert is showing for this swimmer at t
    for a in dash["alerts"]:
        if int(a["person_id"]) != pid or t < float(a["alert_time"]):
            continue
        if a["event"] == "possible_submersion":
            state = "SUBMERSION"
        elif t > float(a.get("end_time", a["alert_time"])) + _keep_s(a, dash) or state in ("SUBMERSION", "DISTRESS"):
            continue
        elif distress.alert_kind(a) == "under_water":   # Rule C: own urgent kind, they had vanished
            state, active = "UNDER_WATER", True
        elif a["event"] == "early_warning":          # Level 1 is a check, not the distress alarm
            state, active = "WARNING", True
        else:
            state, active = "DISTRESS", True
    seen = next((p for p in dash["people"] if p["person_id"] == pid), None)
    if seen and t > seen["last_seen"] + 0.5 and state not in ("SUBMERSION", "DISTRESS") and not active:
        state = "NORMAL" if judged else "NOT_JUDGED"
    return state, risk


# status board order: the most urgent first; someone the rules cannot read comes before someone they read as normal
BOARD_RANK = {"SUBMERSION": 0, "DISTRESS": 1, "UNDER_WATER": 2, "WARNING": 3, "HIGH_RISK": 4, "WATCH": 5,
              "NOT_JUDGED": 6, "NORMAL": 7}


def _board_order(state: str, risk: float) -> tuple[int, float]:
    """Sort key for the status board: state rank, then the highest risk first."""
    return BOARD_RANK[state], -risk


def _in_view(p: dict, t: float) -> bool:
    return p["first_seen"] - 0.5 <= t <= p["last_seen"] + 0.5


def default_time(dash: dict) -> float:
    """Where the replay opens: the first alert, else the moment with the most swimmers in view."""
    duration = float(dash.get("duration_s") or 0.0)
    if dash.get("alerts"):
        return min(min(float(a["alert_time"]) for a in dash["alerts"]), duration)
    grid = [i * 0.5 for i in range(int(duration / 0.5) + 1)]
    return max(grid, key=lambda tt: sum(_in_view(p, tt) for p in dash.get("people") or [])) if grid else 0.0


def level_counts(dash: dict) -> dict:
    """{"aquatic_warning": n, "under_water": n, "aquatic_distress": n, "submersion": n} of the folder's alerts."""
    out = {k: 0 for k in LEVELS}
    for a in dash.get("alerts") or []:
        out[distress.alert_kind(a)] += 1
    return out


URGENCY = ("submersion", "aquatic_distress", "under_water", "aquatic_warning")      # most urgent first
SHORT = {"submersion": "possible submersion", "aquatic_distress": "high-risk distress",
         "under_water": "possibly under water", "aquatic_warning": "swimmer may need help"}


def headline(dash: dict) -> tuple[str, str]:
    """(tone, sentence): the one-line status of a whole video, most urgent first. tone is "urgent", "check" (orange),
    "info" (no alert, but some people were never judged) or "ok"."""
    alerts = dash.get("alerts") or []
    if not alerts:
        return ("info" if distress.never_judged(dash) else "ok"), distress.all_clear_text(dash)
    kind = {id(a): k if k in URGENCY else "aquatic_distress" for a in alerts for k in [distress.alert_kind(a)]}
    kinds = list(kind.values())
    worst = min(kinds, key=URGENCY.index)
    first = min(alerts, key=lambda a: (URGENCY.index(kind[id(a)]), float(a["alert_time"])))
    parts = ", ".join(f"{kinds.count(k)} {SHORT[k]}" for k in URGENCY if k in kinds)
    who = f"Swimmer #{first['person_id']} at {utils.fmt_time(float(first['alert_time']))}"
    if worst == "aquatic_warning":
        return "check", f"**{len(alerts)} alert{'s' if len(alerts) > 1 else ''}** ({parts}). First to check: {who}."
    return "urgent", f"**{len(alerts)} alert{'s' if len(alerts) > 1 else ''}** ({parts}). Go and check {who}."


def alert_cards(folder: Path, data: dict, dash: dict) -> None:
    """One card per alert (most urgent look first), with evidence, pictures and the spoken announcement."""
    path = Path(folder) / "pool_dashboard.json"
    # one swimmer can have several alerts of one kind (two early warnings, a second distress episode): match on time
    events = {}
    for e in data.get("events") or []:
        events.setdefault((e["entity_id"], e["behavior"]), e)
        alert_t = (e.get("metrics") or {}).get("alert_time_s")
        if alert_t is not None:
            events[(e["entity_id"], e["behavior"], round(float(alert_t), 2))] = e
    alerts = dash.get("alerts") or []
    can_play = st.session_state.get("pool_autoplay", True)   # one spoken alert at a time, not all at once
    if not alerts:
        if distress.never_judged(dash):          # not a green all-clear: some people were never looked at
            st.warning(distress.all_clear_text(dash), icon=":material/visibility:")
        else:
            st.success(distress.all_clear_text(dash), icon=":material/pool:")
        return
    for i, a in enumerate(alerts):
        beh = {"possible_submersion": "submersion", "early_warning": "aquatic_warning"}.get(a["event"], "aquatic_distress")
        ev = (events.get((a["person_id"], beh, round(float(a["alert_time"]), 2)))
              or events.get((a["person_id"], beh)) or {})
        kind = distress.alert_kind(a)
        icon, colour, title, level = LEVELS[kind]
        with st.container(key=f"alert_{kind}_{i}"):
            st.markdown(f"**:{colour}[{icon} {title}]**  \n:{colour}-background[{level}]")
            with st.container(horizontal=True):
                st.metric("Swimmer", f"#{a['person_id']}")
                st.metric("Where", distress.where_text({**(ev.get("metrics") or {}), **a}),
                          help="Camera name (when known), zone, and the part of the picture.")
                st.metric("When", utils.fmt_time(float(a["alert_time"])),
                          help=f"Behaviour began at {utils.fmt_time_precise(float(a['start_time']))}; "
                               f"alert raised at {utils.fmt_time_precise(float(a['alert_time']))}.")
                if kind == "under_water":                 # a fixed Level 1 score would read like a probability
                    st.metric("Out of sight", f"{float(a['alert_time']) - float(a['start_time']):.0f} s",
                              help="From when they vanished to when somebody was seen again at that spot.")
                else:
                    st.metric("Risk", f"{float(a['risk_score']):.0%}")
            st.markdown("  \n".join(f"✓ {distress.EVIDENCE_TEXT.get(k, k)}" for k in a["evidence"]))
            ann = ev.get("announcement") or {}
            audio = Path(folder) / ann["audio"] if ann.get("audio") else None
            if audio is not None and audio.is_file():
                st.caption(f":material/volume_up: “{ann.get('text', '')}”")
                key = f"played::{folder}::{path.stat().st_mtime}::{i}"
                first = not st.session_state.get(key, False)
                play = first and can_play                  # only the first unplayed alert speaks on this view
                if play:
                    can_play = False
                # own alt text per card: two alerts on one swimmer can have byte-identical audio (same sentence), which
                # Streamlit would reject as a duplicate element (st.audio has no key) and the whole page would fail
                st.audio(str(audio), format="audio/wav", autoplay=play,
                         alt=f"Spoken alert {i + 1} of {len(alerts)}: {ann.get('text', '')}")
                if play:
                    st.session_state[key] = True
            with st.expander("View evidence"):
                snap = Path(folder) / ev["snapshot"] if ev.get("snapshot") else None
                plot = Path(folder) / ev["speed_plot"] if ev.get("speed_plot") else None
                if snap is not None and snap.is_file():
                    st.image(str(snap), caption="Snapshot at the alert", alt=f"Snapshot of the alert on swimmer {a['person_id']}")
                if plot is not None and plot.is_file():
                    st.image(str(plot), caption="Distress risk over time", alt="Risk of this swimmer over time")
                if ev.get("evidence"):
                    st.write(ev["evidence"])


def status_board(dash: dict, t: float) -> None:
    """The swimmers in view at time t (plus anyone in DISTRESS / SUBMERSION who was just lost), with their state."""
    alert_ids = {int(a["person_id"]) for a in dash.get("alerts") or []}
    shown = [p for p in dash.get("people") or [] if _in_view(p, t) or
             (p["person_id"] in alert_ids and _state_at(t, p["person_id"], dash)[0] in ALERT_STATES)]
    shown.sort(key=lambda p: _board_order(*_state_at(t, p["person_id"], dash)))
    with st.container(key="board_head"):
        st.markdown(f"**Status at {utils.fmt_time_precise(t)}** · {len(shown)} swimmer(s) in view")
    with st.container(key="board_scroll"):          # ui_theme: scrolls inside the board; the cards fill it as a grid
        if not shown:
            st.info("Nobody is in view at this moment. Move the slider.")
            return
        with st.container(horizontal=True, key="swimgrid"):
            for p in shown:
                state, risk = _state_at(t, p["person_id"], dash)
                with st.container(key=f"swim_{state}_{p['person_id']}"):
                    st.markdown(f"**{p['name']}**  \n{STATE_BADGE[state]}  \n"
                                + ("_lost from view_" if not _in_view(p, t) else
                                   f"_{NOT_JUDGED_TEXT}_" if state == "NOT_JUDGED" else f"risk {risk:.0%}"))


def risk_chart(dash: dict, t: float | None = None, height: int = 240) -> None:
    """Distress risk of every swimmer over time, the alert level (dashed red) and the replay time (grey)."""
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        return
    duration = float(dash.get("duration_s") or 0.0)
    rows = [{"time (s)": tt, "risk": r, "swimmer": p["name"]}
            for p in dash.get("people") or [] for tt, r in (dash["risk"].get(str(p["person_id"])) or [])]
    if not rows:
        st.caption("No risk data in this folder.")
        return
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("time (s):Q", scale=alt.Scale(domain=[0, duration])),
        y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("swimmer:N", legend=alt.Legend(orient="bottom")), tooltip=["swimmer", "time (s)", "risk"])
    chart += alt.Chart(pd.DataFrame({"risk": [dash["thresholds"]["warning"]]})).mark_rule(
        strokeDash=[6, 4], color="#e67e22").encode(y="risk:Q")
    chart += alt.Chart(pd.DataFrame({"risk": [dash["thresholds"]["alert"]]})).mark_rule(
        strokeDash=[6, 4], color="#c0392b").encode(y="risk:Q")
    if t is not None:
        chart += alt.Chart(pd.DataFrame({"time (s)": [t]})).mark_rule(color="#888").encode(x="time (s):Q")
    chart = chart.properties(background="transparent").configure_view(strokeWidth=0)     # sits on the glass panel
    st.altair_chart(chart, height=height, alt="Distress risk of every swimmer over time")
    st.caption(f"Dashed lines: early warning {dash['thresholds']['warning']:.0%} (orange), distress alarm "
               f"{dash['thresholds']['alert']:.0%} (red). Grey line: the replay time.")


def behaviour_timelines(dash: dict, expanded: bool = False) -> None:
    """What the system noticed about each swimmer, in order."""
    timelines = {k: v for k, v in (dash.get("timeline") or {}).items() if v}
    if not timelines:
        st.caption("No behaviour notes in this folder.")
        return
    for pid, items in timelines.items():
        st.markdown(f"**Swimmer #{pid}**  \n" + "  \n".join(
            f"`{utils.fmt_time_precise(x['t'])}` {x['note']}" for x in items))


def render_pool_panel(folder: Path, data: dict) -> None:
    """All lifeguard pieces in one column (the older layout; app.py now places the pieces itself)."""
    dash = load_dashboard(folder)
    if dash is None:
        return
    st.subheader("Lifeguard dashboard")
    alert_cards(folder, data, dash)
    st.toggle("Auto-play spoken alerts", value=True, key="pool_autoplay",
              help="Plays the alarm and the spoken alert once when new results appear, on this device's speaker.")
    duration = float(dash.get("duration_s") or 0.0)
    if duration > 0 and dash.get("people"):
        t = st.slider("Replay the pool at time (s)", 0.0, duration, default_time(dash), step=0.5,
                      key=f"pool_t::{folder}")
        status_board(dash, t)
        risk_chart(dash, t)
    with st.expander("Behaviour timelines (what the system noticed, in order)", expanded=bool(dash.get("alerts"))):
        behaviour_timelines(dash)
