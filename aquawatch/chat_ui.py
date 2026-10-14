"""The floating "Ask AquaWatch" chat of the dashboard (app.py, both views).

A round button stays at the bottom-right of every page. It opens a full-height glass side drawer (st.dialog with
position="right") with the conversation about whatever is open:
  * the multi-camera site's event index (site_ui sets the context), or
  * the results folder shown in the single-camera view (app.py sets the context; ps_conversation.folder_conversation).
The drawer is laid out like a messaging app: what the chat is about and "New chat" at the top, the conversation in
the middle (it scrolls and stays on the newest answer; the user's questions on the right in blue, AquaWatch's answers
on the left in white, each with a status chip and its sources), then the suggestions and the input at the bottom.
When AquaWatch asks "which camera covers the deep end?", one button per camera answers it in one click.
Answers come from ps_conversation (grounded: camera, time, evidence). A visual answer found by CLIP frame ranking gets a
"Check with YOLOE" button: the same question again with the YOLOE retrieve-then-detect method, which draws a box or
says "no confident match" (about 10 s on this laptop's CPU). The chat history is kept per site / folder for the session.
The chat's text model takes about 30 s to load on this laptop: warm_up() loads it in the background when the
dashboard starts, so the first question does not wait.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import streamlit as st

CONTEXT_KEY = "chat_context"
HISTORY_SHOWN = 20                       # turns drawn in the drawer (older ones stay in the session)
HISTORY_HEIGHT = 440                     # px of the scrolling conversation in the drawer
SITE_EXAMPLES = ["What happened?", "Who is drowning?", "Which camera saw the highest risk?",
                 "What happened at the deep end?", "Is there a child in a blue life jacket?"]
FOLDER_EXAMPLES = ["What happened?", "Who is drowning?", "Which swimmer had the highest risk?",
                   "How many swimmers were in view at 00:10?", "Show the evidence"]


def set_context(kind: str, path: str | Path, title: str, examples: list[str] | None = None) -> None:
    """What the chat talks about: kind "site" (an event index .sqlite3) or "folder" (one results folder)."""
    st.session_state[CONTEXT_KEY] = {"kind": kind, "path": str(path), "title": title,
                                     "examples": list(examples or (SITE_EXAMPLES if kind == "site" else FOLDER_EXAMPLES))}


@st.cache_resource(show_spinner=False)
def _warm_up_thread() -> threading.Thread | None:
    """Start loading the chat's text model in a background thread, once per dashboard process. Not under pytest:
    the tests must not load a 300 MB model they do not use."""
    if "pytest" in sys.modules:
        return None
    import chat_router
    thread = threading.Thread(target=chat_router.warm_up, name="aquawatch-chat-warm-up", daemon=True)
    thread.start()
    return thread


def warm_up() -> None:
    """Load the chat's model in the background now (the dashboard calls this at start; it runs only once)."""
    try:
        _warm_up_thread()
    except Exception:                                 # never let a warm-up problem break the page
        pass


def _engine_key(ctx: dict) -> str:
    """The session key of the engine for this site / results folder VERSION (index or events.json mtime)."""
    if ctx["kind"] == "site":
        import site_ui
        return site_ui.site_engine_key(ctx["path"])
    events = Path(ctx["path"]) / "events.json"
    return f"folder_engine::{ctx['path']}::{events.stat().st_mtime if events.is_file() else 0}"


def _engine(ctx: dict):
    """One conversation engine per site / results folder version, kept in the session (it remembers a pending
    clarification). A folder is re-indexed when its events.json changes, a site when its index changes."""
    key = _engine_key(ctx)
    if ctx["kind"] == "site":
        if key not in st.session_state:
            from ps_conversation import SiteConversation
            st.session_state[key] = SiteConversation(ctx["path"])
        return st.session_state[key]
    if key not in st.session_state:
        from ps_conversation import folder_conversation
        st.session_state[key] = folder_conversation(ctx["path"])
    return st.session_state[key]


def _is_clip_answer(result: dict) -> bool:
    """True for an answer ranked by CLIP alone (it can be checked with the YOLOE detector)."""
    return any(str(c.get("event_state") or "").startswith("open-vocabulary match") for c in result.get("citations") or [])


def _picked(key: str) -> None:
    """A suggestion chip was clicked: ask it, and clear the chip so the same one can be asked again later."""
    st.session_state["_chat_ask"] = st.session_state.get(key)
    st.session_state[key] = None


@st.dialog("Ask AquaWatch", position="right", width="medium", icon=":material/forum:")
def _drawer() -> None:
    """The chat drawer. Widgets here rerun only the drawer."""
    import site_ui
    import ui_helpers as ui
    ctx = st.session_state.get(CONTEXT_KEY)
    if not ctx:
        st.info("Run an analysis or open a results folder first: the chat answers about what is on the page.",
                icon=":material/info:")
        return
    try:
        engine = _engine(ctx)
    except Exception as exc:                          # a missing index must not break the page
        st.error(f"The chat is not available here ({type(exc).__name__}: {exc}).")
        return
    history = st.session_state.setdefault(f"chat::{_engine_key(ctx)}", [])   # old answers stay with old results

    with st.container(horizontal=True, vertical_alignment="center", key="chat_context"):
        st.badge("Site" if ctx["kind"] == "site" else "Video",
                 icon=":material/videocam:" if ctx["kind"] == "site" else ":material/movie:", color="blue")
        st.markdown(f"**{ui.md_escape(ctx['title'])}**", width="stretch")
        if st.button("New chat", key=f"chat_new::{ctx['path']}", type="tertiary", icon=":material/edit_square:",
                     help="Start an empty conversation (saved places are kept)."):
            history.clear()
            engine.pending = None
            if hasattr(engine, "pending_place"):
                engine.pending_place = None
    st.caption("Answers come only from the analysis (camera, time, evidence). Descriptions are matched with CLIP; "
               "faces, gender or age are never searched.")

    conversation = st.container(height=HISTORY_HEIGHT, autoscroll=True, border=False, key="chat_history")
    waiting = st.container()                          # "which camera?" quick answers, under the conversation
    pick_key = f"chat_pick::{ctx['path']}"
    st.pills("Suggestions", ctx["examples"], key=pick_key, on_change=_picked, args=(pick_key,),
             label_visibility="collapsed")
    typed = st.chat_input("Ask about the footage, e.g. what happened between 00:10 and 00:15?",
                          key=f"chat_in::{ctx['path']}", submit_mode="disable")

    question, method = typed or None, None
    for source in ("_chat_ask", "_chat_reply"):      # a suggestion chip, or a camera button
        picked = st.session_state.pop(source, None)
        if picked:
            question = picked
    check = st.session_state.pop("_chat_check", None)  # "Check with YOLOE" pressed on an earlier answer
    if check:
        question, method = check, "ground_yoloe"
    if question:
        with conversation, st.spinner("Checking the frames with YOLOE ..." if method else "Looking it up ..."):
            result = engine.answer(question, visual_method=method)
        history.append({"q": question + ("  (checked with YOLOE)" if method else ""), "r": result})

    with waiting:
        if engine.pending:
            with st.container(key="chat_pending"):
                st.markdown(f":material/help: **Waiting for your answer to:** {ui.md_escape(engine.pending)}")
                if getattr(engine, "pending_place", None) and ctx["kind"] == "site":
                    st.caption("Which camera is it? One click saves it (you can change it later).")
                    with st.container(horizontal=True):
                        for cid, row in engine.cameras.items():
                            if st.button(str(row.get("name") or cid.upper()), key=f"chat_cam::{ctx['path']}::{cid}",
                                         type="secondary", icon=":material/videocam:"):
                                st.session_state["_chat_reply"] = cid
                                st.rerun(scope="fragment")

    with conversation:
        if not history:
            st.markdown("**Ask about the footage**", text_alignment="center")
            st.caption("Every answer names the camera, the time and the evidence. Pick a suggestion below or type "
                       "a question.", text_alignment="center")
        for n, turn in enumerate(history[-HISTORY_SHOWN:]):
            site_ui._show_answer(turn["q"], turn["r"])
            if _is_clip_answer(turn["r"]) and "(checked with YOLOE)" not in turn["q"]:
                if st.button("Check with YOLOE (about 10 s)", key=f"chat_yoloe::{ctx['path']}::{n}",
                             icon=":material/center_focus_strong:", type="secondary",
                             help="Runs the YOLOE open-vocabulary detector on the frames CLIP picked: a box when the "
                                  "thing is found, 'no confident match' when it is not."):
                    st.session_state["_chat_check"] = turn["q"]
                    st.rerun(scope="fragment")


def render_fab() -> None:
    """The floating button (call it once, at the end of the page; ui_theme.py pins it bottom-right)."""
    with st.container(key="ask_fab"):
        if st.button("Ask AquaWatch", icon=":material/forum:", type="primary", key="ask_fab_button",
                     help="Ask questions about the open video or site"):
            _drawer()
