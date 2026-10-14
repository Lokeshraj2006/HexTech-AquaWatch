"""The look of the AquaWatch dashboard: "glassmorphism" in neutral grays, in a light and a dark version, like a Mac
app: see-through frosted gray panels over a plain neutral background (no colour patches; colour is kept for meaning:
the blue action, the alert colours).

Colours, fonts and corner radius of Streamlit's own widgets are set in .streamlit/config.toml, which has a light AND a
dark theme ([theme.light], [theme.dark]). This module draws what the theme cannot: the background, the glass sidebar
and panels, the tab bar, the chat bubbles and the floating chat button, with one set of CSS variables per mode.
Which mode is on is read in the browser (a tiny script compares the app's own background colour), so the glass always
matches Streamlit's widgets, also right after a switch. The Light / Dark switch at the top of the sidebar stores the
choice the same way Streamlit's own menu (⋮ > Settings) does, then reloads the page once.

A panel is marked by its container key: glass("alerts") is st.container(key="glass_alerts"), which Streamlit gives
the class "st-key-glass_alerts", so no Streamlit-internal class name is needed.
"""
from __future__ import annotations

import html
import json
from urllib.parse import quote

import streamlit as st

MODES = {"Light": ":material/light_mode:", "Dark": ":material/dark_mode:"}
# where Streamlit 1.65 keeps the chosen theme in the browser (a JavaScript expression: it depends on the page path)
THEME_STORAGE_KEY = '"stActiveTheme-" + window.location.pathname + "-v2"'

# the wave mark of the sidebar brand, drawn by CSS: an inline SVG in st.html is removed by Streamlit's HTML cleaner,
# and a raw "<svg" even inside the <style> makes the cleaner drop the whole stylesheet, so it is URL-encoded
_WAVE = 'url("data:image/svg+xml,{}")'.format(quote(
    "<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='white' "
    "stroke-width='2.2' stroke-linecap='round'><path d='M2 9c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2.5 2 5 2'/>"
    "<path d='M2 15c2.5 0 2.5-2 5-2s2.5 2 5 2 2.5-2 5-2 2.5 2 5 2'/></svg>", safe=""))

CSS = """
<style>
/* ======== the two modes: everything below uses these variables (neutral grays only) ======== */
/* glass = 80 % see-through (20 % opacity), drawn by its hairline edge, a top highlight and a tight shadow, like macOS */
:root {
  /* gray water light (tools/make_wallpaper.py, served from static/): something for the glass to show through,
     like a Mac desktop picture; gray only */
  --aw-wallpaper: url("app/static/wallpaper_light.jpg") center / cover no-repeat, #e4e4e9;
  --aw-glass: rgba(255, 255, 255, 0.20);
  --aw-glass-2: rgba(255, 255, 255, 0.20);       /* cards inside the glass (alerts, swimmers) */
  --aw-field: rgba(255, 255, 255, 0.72);         /* text fields and selects: like a Mac text field */
  --aw-glass-hover: rgba(255, 255, 255, 0.55);
  --aw-track: rgba(118, 118, 128, 0.12);         /* segmented control / tab bar track */
  --aw-thumb: #ffffff;                           /* the selected segment */
  --aw-tab-on: #ffffff;
  --aw-dialog: rgba(246, 246, 248, 0.20);
  --aw-bubble: rgba(255, 255, 255, 0.70);
  --aw-edge: rgba(0, 0, 0, 0.09);
  --aw-hairline: rgba(0, 0, 0, 0.08);
  --aw-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.70);
  --aw-text: #1d1d1f;
  --aw-text-2: #6e6e73;
  --aw-text-3: #86868b;
  --aw-blue: #0071e3;
  --aw-push: #0071e3;                            /* the blue push button */
  --aw-blur: blur(18px) saturate(140%);           /* light enough that the water shows through */
  --aw-blur-small: blur(12px) saturate(130%);
  --aw-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.03), 0 1px 2px rgba(0, 0, 0, 0.05), 0 4px 14px rgba(0, 0, 0, 0.05);
  --aw-shadow-lg: 0 0 0 0.5px rgba(0, 0, 0, 0.05), 0 2px 6px rgba(0, 0, 0, 0.06), 0 16px 44px rgba(0, 0, 0, 0.10);
  --aw-thumb-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.14);
}
:root[data-aw-theme="dark"] {
  --aw-wallpaper: url("app/static/wallpaper_dark.jpg") center / cover no-repeat, #141416;
  --aw-glass: rgba(72, 72, 74, 0.20);
  --aw-glass-2: rgba(255, 255, 255, 0.04);
  --aw-field: rgba(255, 255, 255, 0.07);
  --aw-glass-hover: rgba(255, 255, 255, 0.10);
  --aw-track: rgba(118, 118, 128, 0.24);
  --aw-thumb: #636366;
  --aw-tab-on: #636366;
  --aw-dialog: rgba(40, 40, 42, 0.22);
  --aw-bubble: rgba(255, 255, 255, 0.08);
  --aw-edge: rgba(255, 255, 255, 0.09);
  --aw-hairline: rgba(255, 255, 255, 0.08);
  --aw-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.06);
  --aw-text: #f5f5f7;
  --aw-text-2: #98989d;
  --aw-text-3: #8e8e93;
  --aw-blue: #0a84ff;
  --aw-push: #0a84ff;
  --aw-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.30), 0 1px 2px rgba(0, 0, 0, 0.30), 0 6px 18px rgba(0, 0, 0, 0.30);
  --aw-shadow-lg: 0 0 0 0.5px rgba(0, 0, 0, 0.40), 0 2px 6px rgba(0, 0, 0, 0.35), 0 18px 48px rgba(0, 0, 0, 0.50);
  --aw-thumb-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.25), 0 1px 3px rgba(0, 0, 0, 0.35);
}

/* ======== background (only the image: the colour underneath stays Streamlit's, so the mode can be read) ======== */
/* the wallpaper sits on its own fixed layer behind the app: it never repaints when the page scrolls or a video
   plays (a fixed background-attachment did, which made the blurred glass flicker) */
.stApp { isolation: isolate; }
.stApp::before { content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background: var(--aw-wallpaper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; padding-bottom: 6.5rem; max-width: 1480px; }

/* ======== sidebar: one floating glass panel ======== */
[data-testid="stSidebar"] { background: transparent !important; border-right: none !important; box-shadow: none; }
[data-testid="stSidebar"] > div:first-child {
  transform: translateZ(0);                       /* its own compositing layer: no flicker */
  background: var(--aw-glass) !important;
  -webkit-backdrop-filter: var(--aw-blur); backdrop-filter: var(--aw-blur);
  border: 1px solid var(--aw-edge); border-radius: 18px;
  margin: 10px 0 10px 10px; height: calc(100% - 20px);
  box-shadow: var(--aw-shadow-lg), var(--aw-highlight);
}
[data-testid="stSidebarUserContent"] { padding-top: 0.2rem; }
.aw-brand { display: flex; align-items: center; gap: 0.7rem; }
.aw-logo {
  width: 40px; height: 40px; border-radius: 12px; flex: none;
  background: __WAVE__ center / 24px 24px no-repeat, linear-gradient(145deg, #5ac8fa 0%, #0a84ff 55%, #0058c4 100%);
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.45);
}
.aw-name { font-weight: 700; font-size: 1.12rem; letter-spacing: -0.01em; line-height: 1.15; color: var(--aw-text); }
.aw-sub { font-size: 0.78rem; color: var(--aw-text-2); line-height: 1.2; }
.st-key-aw_modes { background: var(--aw-track); border-radius: 9px; padding: 2px; gap: 2px; }
.st-key-aw_modes button { min-height: 1.75rem; padding: 0.1rem 0.45rem; border-radius: 7px; box-shadow: none !important; }
.st-key-aw_modes [data-testid="stBaseButton-primary"]:not(:disabled) {
  background: var(--aw-thumb) !important; color: var(--aw-text) !important; box-shadow: var(--aw-thumb-shadow) !important;
}
.st-key-aw_modes [data-testid="stBaseButton-primary"] p, .st-key-aw_modes [data-testid="stBaseButton-primary"] span {
  color: var(--aw-text) !important; }
.aw-section { font-size: 0.78rem; font-weight: 600; color: var(--aw-text-3); margin: 0.95rem 0 0.05rem 0.15rem; }

/* ======== glass panels (containers with a key that starts with "glass") ======== */
[class*="st-key-glass"] {
  transform: translateZ(0);                       /* its own compositing layer: no flicker */
  background: var(--aw-glass);
  -webkit-backdrop-filter: var(--aw-blur); backdrop-filter: var(--aw-blur);
  border: 1px solid var(--aw-edge); border-radius: 16px;
  box-shadow: var(--aw-shadow), var(--aw-highlight);
  padding: 1.05rem 1.2rem;
}
[class*="st-key-glass"] [class*="st-key-glass"] {      /* a panel inside a panel: no second blur */
  -webkit-backdrop-filter: none; backdrop-filter: none; background: var(--aw-glass-2); box-shadow: none;
}
.st-key-glass_header h1, .st-key-glass_site_header h1 { font-size: 1.85rem; letter-spacing: -0.02em; padding: 0; }
.aw-overline { font-size: 0.82rem; font-weight: 600; color: var(--aw-text-2); }

/* ======== the one-line status of a result ======== */
[class*="st-key-headline_"] {
  background: var(--aw-glass); border-radius: 12px; padding: 0.75rem 1.1rem;
  border: 1px solid var(--aw-edge); border-left-width: 3px !important; box-shadow: var(--aw-shadow);
  -webkit-backdrop-filter: var(--aw-blur-small); backdrop-filter: var(--aw-blur-small);
  box-shadow: var(--aw-highlight);
}
.st-key-headline_ok { border-left-color: #34c759 !important; }
.st-key-headline_check { border-left-color: #ff9f0a !important; }
.st-key-headline_urgent { border-left-color: #ff3b30 !important; }
.st-key-headline_info { border-left-color: #0a84ff !important; }

/* ======== metric tiles ======== */
[data-testid="stMetric"] {
  background: var(--aw-glass-2);
  -webkit-backdrop-filter: var(--aw-blur-small); backdrop-filter: var(--aw-blur-small);
  border: 1px solid var(--aw-edge) !important; border-radius: 12px !important;
  box-shadow: var(--aw-shadow), var(--aw-highlight);
  padding: 0.75rem 0.95rem !important;
}
[class*="st-key-glass"] [data-testid="stMetric"] {
  background: transparent; border: none !important; box-shadow: none; padding: 0 !important;
  -webkit-backdrop-filter: none; backdrop-filter: none;
}
[data-testid="stMetricLabel"] { color: var(--aw-text-2); }
[data-testid="stMetricLabel"] p { font-size: 0.8rem; font-weight: 500; }
[data-testid="stMetricValue"] { letter-spacing: -0.02em; font-weight: 600; font-variant-numeric: tabular-nums; }
.st-key-levels [data-testid="stMetricValue"] { font-size: 1.35rem; }
.st-key-levels [data-testid="stMetricLabel"] p { text-transform: none; letter-spacing: 0; font-size: 0.8rem; }
[class*="st-key-alert_"] [data-testid="stMetricValue"] { font-size: 1.05rem; line-height: 1.3; white-space: normal; }
[data-testid="stMainBlockContainer"] h1 { font-size: 1.8rem; line-height: 1.2; font-weight: 700; }
[data-testid="stMainBlockContainer"] h3 { font-size: 1.25rem; font-weight: 600; }
.stApp h1 { letter-spacing: -0.02em; }
.aw-foot { font-size: 0.75rem; color: var(--aw-text-2); margin-top: 1.4rem; line-height: 1.4; }

/* ======== alert cards and swimmer chips (pool_ui) ======== */
[class*="st-key-alert_"] {
  background: var(--aw-glass-2); border: 1px solid var(--aw-edge); border-radius: 12px;
  padding: 0.85rem 1rem; border-left-width: 3px !important; box-shadow: var(--aw-highlight);
}
[class*="st-key-alert_aquatic_warning"] { border-left-color: #ff9f0a !important; }
[class*="st-key-alert_under_water"] { border-left-color: #0a84ff !important; }
[class*="st-key-alert_aquatic_distress"] { border-left-color: #ff3b30 !important; }
[class*="st-key-alert_submersion"] { border-left-color: #af52de !important; }
[class*="st-key-swim_"] {
  background: var(--aw-glass-2); border: 1px solid var(--aw-edge); border-radius: 10px;
  padding: 0.45rem 0.65rem; border-top-width: 2px !important; box-shadow: var(--aw-highlight);
}
[class*="st-key-swim_NORMAL"] { border-top-color: #34c759 !important; }
[class*="st-key-swim_WATCH"] { border-top-color: #0a84ff !important; }
[class*="st-key-swim_HIGH_RISK"], [class*="st-key-swim_WARNING"] { border-top-color: #ff9f0a !important; }
[class*="st-key-swim_DISTRESS"], [class*="st-key-swim_UNDER_WATER"] { border-top-color: #ff3b30 !important; }
[class*="st-key-swim_SUBMERSION"] { border-top-color: #af52de !important; }
[class*="st-key-swim_NOT_JUDGED"] { border-top-color: #8e95a3 !important; }
[class*="st-key-swim_"] p { font-size: 0.86rem; line-height: 1.35; }
[class*="st-key-swim_"] span { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }

/* ======== the replay: video and swimmer status side by side, the same height ======== */
/* (the same on the multi-camera page: site_row = the camera video and the 'Active swimmers' list) */
.st-key-replay_row, .st-key-site_row { display: grid !important; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
  gap: 1rem; align-items: stretch; }
.st-key-replay_row > div, .st-key-site_row > div { min-width: 0; width: auto !important; }
.st-key-glass_board, .st-key-glass_site_swimmers { position: relative; height: 100%; min-height: 360px;
  box-sizing: border-box; }
.st-key-board_head p, .st-key-site_swim_head p { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.st-key-board_scroll, .st-key-site_swim_scroll { position: absolute !important; left: 1.2rem; right: 0.6rem;
  top: 3.4rem; bottom: 1rem; width: auto !important; height: auto !important; overflow-y: auto; padding-right: 0.6rem; }
.st-key-swimgrid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 0.6rem; }
.st-key-swimgrid > div { width: auto !important; min-width: 0; }
/* the alert cards side by side, as many as fit */
.st-key-alertgrid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 0.9rem; align-items: start; }
.st-key-alertgrid > div { width: auto !important; min-width: 0; }
.st-key-alertgrid > div:only-child { grid-column: 1 / -1; }
@media (max-width: 900px) {                      /* narrow screens: one column, the board a fixed height */
  .st-key-replay_row, .st-key-site_row { grid-template-columns: minmax(0, 1fr); }
  .st-key-glass_board, .st-key-glass_site_swimmers { height: 460px; }
}

/* ======== tabs: a segmented control, like macOS ======== */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--aw-track); border: none; border-radius: 9px;
  padding: 2px; gap: 2px; width: fit-content; max-width: 100%;
}
[data-testid="stTabs"] [role="tab"] { border-radius: 7px; padding: 0.3rem 0.95rem; height: auto; margin: 0; }
[data-testid="stTabs"] [role="tab"] p { color: var(--aw-text); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: var(--aw-tab-on); box-shadow: var(--aw-thumb-shadow); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [role="tab"] > div:not([data-testid="stMarkdownContainer"]) { display: none; }  /* the blue bar */

/* ======== inputs, expanders, alerts, tables, media, code: see-through too ======== */
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="textarea"], .stApp [data-testid="stNumberInputContainer"] {
  background-color: var(--aw-field) !important; border-color: var(--aw-edge) !important;
}
.stApp [data-baseweb="input"] input, .stApp [data-baseweb="base-input"] input { background-color: transparent !important; }
[data-testid="stExpander"] details { background: var(--aw-glass-2); border: 1px solid var(--aw-edge) !important; border-radius: 10px; }
[data-testid="stAlertContainer"] { border-radius: 10px; -webkit-backdrop-filter: var(--aw-blur-small); backdrop-filter: var(--aw-blur-small); }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
[data-testid="stVideo"], [data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stCode"] pre { border-radius: 8px; }
[data-testid="stBaseButton-secondary"] {
  background: var(--aw-thumb) !important; border: 1px solid var(--aw-edge) !important;
  box-shadow: 0 0.5px 1.5px rgba(0, 0, 0, 0.10) !important;
}
[data-testid="stBaseButton-secondary"]:hover { filter: brightness(0.98); }
/* segmented control: a gray track, the chosen segment a raised thumb (not a blue outline) */
[data-testid="stButtonGroup"] div:has(> button[data-variant="segmented_control"]) {
  background: var(--aw-track); border-radius: 9px; padding: 2px; gap: 2px !important;
}
button[data-variant="segmented_control"] {
  background: transparent !important; border: none !important; border-radius: 7px !important;
  color: var(--aw-text) !important; box-shadow: none !important; min-height: 1.9rem; margin: 0 !important;
}
button[data-variant="segmented_control"]:hover { background: var(--aw-glass-hover) !important; }
button[data-variant="segmented_control"][aria-checked="true"] {
  background: var(--aw-thumb) !important; box-shadow: var(--aw-thumb-shadow) !important; font-weight: 500;
}
button[data-variant="segmented_control"] p { color: var(--aw-text) !important; }
/* in the sidebar: compact, so two or three choices stay on one line */
[data-testid="stSidebar"] button[data-variant="segmented_control"] { padding: 0.2rem 0.55rem !important; min-height: 1.8rem; }
[data-testid="stSidebar"] button[data-variant="segmented_control"] p { font-size: 0.85rem; }
[data-testid="stSidebar"] [data-testid="stButtonGroup"] div:has(> button[data-variant="segmented_control"]) {
  flex-wrap: nowrap !important; }
[data-testid="stSidebar"] button[data-variant="segmented_control"] { flex: 1 1 auto; min-width: 0; }
/* the page switch (Single camera / Multi-camera): a sidebar list like Finder's, the chosen row a gray highlight */
.st-key-view [data-testid="stButtonGroup"] div:has(> button[data-variant="segmented_control"]) {
  background: transparent; flex-direction: column; flex-wrap: nowrap !important; padding: 0; gap: 2px !important; }
.st-key-view button[data-variant="segmented_control"] {
  justify-content: flex-start !important; width: 100%; padding: 0.35rem 0.6rem !important; min-height: 2rem; }
.st-key-view button[data-variant="segmented_control"][aria-checked="true"] {
  background: var(--aw-track) !important; box-shadow: none !important; font-weight: 600; }
.st-key-view button[data-variant="segmented_control"] p { font-size: 0.9rem; }
/* pills (chat suggestions): soft gray capsules */
button[data-variant="pills"] {
  background: var(--aw-track) !important; border: none !important; color: var(--aw-text) !important;
}
button[data-variant="pills"]:hover { filter: brightness(0.96); }
/* the blue push button: flat, a fine shadow (no glow) */
[data-testid="stBaseButton-primary"]:not(:disabled) {
  background: var(--aw-push) !important; border: 1px solid rgba(0, 0, 0, 0.06) !important;
  box-shadow: inset 0 0.5px 0 rgba(255, 255, 255, 0.25), 0 0.5px 1.5px rgba(0, 0, 0, 0.20) !important;
}
[data-testid="stBaseButton-primary"]:not(:disabled):hover { filter: brightness(1.06); }

/* ======== the floating "Ask AquaWatch" button ======== */
.st-key-ask_fab { position: fixed; bottom: 1.4rem; right: 1.6rem; z-index: 990; width: auto; }
.st-key-ask_fab button {
  border-radius: 999px; padding: 0.55rem 1.15rem; background: var(--aw-push) !important;
  border: 1px solid rgba(0, 0, 0, 0.06) !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.16), 0 8px 22px rgba(0, 0, 0, 0.10) !important;
}

/* ======== dialogs (the chat drawer, the new-site form) ======== */
[data-testid="stDialog"] { background: rgba(10, 20, 40, 0.16) !important; }
[data-testid="stDialog"] > div {
  background: var(--aw-dialog) !important;
  -webkit-backdrop-filter: blur(44px) saturate(180%); backdrop-filter: blur(44px) saturate(180%);
  border: 1px solid var(--aw-edge) !important; box-shadow: var(--aw-shadow-lg) !important;
}

/* ======== chat: the user's questions on the right in blue, AquaWatch's answers on the left ======== */
[data-testid="stChatMessage"] { background: transparent !important; padding: 0.2rem 0; gap: 0.55rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 82%; margin-left: auto;
  background: linear-gradient(180deg, rgba(43, 140, 255, 0.92), rgba(0, 113, 227, 0.92));
  border-radius: 18px 18px 5px 18px; padding: 0.5rem 0.9rem; box-shadow: 0 4px 14px rgba(0, 113, 227, 0.30);
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p { color: #ffffff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
  background: var(--aw-bubble); border: 1px solid var(--aw-edge);
  border-radius: 18px 18px 18px 5px; padding: 0.65rem 0.95rem; box-shadow: var(--aw-highlight);
}
[data-testid="stChatInput"] { border-radius: 18px; background: var(--aw-glass-2); border: 1px solid var(--aw-edge); }
.st-key-chat_pending {
  background: rgba(255, 159, 10, 0.16); border: 1px solid rgba(255, 159, 10, 0.40); border-radius: 14px;
  padding: 0.6rem 0.8rem;
}
.aw-empty { text-align: center; padding: 1.6rem 0.5rem 0.6rem; color: var(--aw-text-2); }
.aw-empty .aw-empty-title { font-weight: 700; font-size: 1.05rem; color: var(--aw-text); margin-top: 0.5rem; }
</style>
""".replace("__WAVE__", _WAVE)

# Reads the mode Streamlit is showing (light or dark: the brightness of the app's own background colour) and marks
# <html data-aw-theme="...">, so the glass variables above follow it, also right after a switch in the menu.
WATCH_JS = """<script>
(function () {
  function bright(rgb) {
    var m = (rgb || "").match(/[0-9.]+/g);
    if (!m || m.length < 3) return 1;
    return (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
  }
  function sync() {
    var app = document.querySelector(".stApp");
    if (!app) return;
    var mode = bright(getComputedStyle(app).backgroundColor) < 0.5 ? "dark" : "light";
    if (document.documentElement.getAttribute("data-aw-theme") !== mode) {
      document.documentElement.setAttribute("data-aw-theme", mode);
    }
  }
  sync();
  if (!window.__awThemeWatch) { window.__awThemeWatch = setInterval(sync, 400); }
})();
</script>"""


def apply() -> None:
    """Load the glass theme (call once per page run, before anything is drawn)."""
    st.html(CSS)
    st.html(WATCH_JS, unsafe_allow_javascript=True)
    switch = st.session_state.pop("_aw_switch_mode", None)
    if switch in MODES:                              # the sidebar switch was used: store it like Streamlit does
        st.html(f"<script>try {{ localStorage.setItem({THEME_STORAGE_KEY}, {json.dumps(json.dumps(switch))}); }}"
                " catch (e) {} window.location.reload();</script>", unsafe_allow_javascript=True)


def current_mode() -> str:
    """"Dark" or "Light": the mode the browser shows now (Streamlit reports it with every run)."""
    try:
        return "Dark" if st.context.theme.type == "dark" else "Light"
    except Exception:
        return "Light"


def brand(subtitle: str = "Lifeguard assistant") -> None:
    """The sidebar brand (wave mark, name, one line) with the Light / Dark buttons beside it. Plain buttons, not a
    switch widget: a widget whose default follows the theme changed the theme by itself once (2026-10-13)."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.html(f'<div class="aw-brand"><div class="aw-logo"></div><div><div class="aw-name">AquaWatch</div>'
                f'<div class="aw-sub">{html.escape(subtitle)}</div></div></div>', width="stretch")
        now = current_mode()
        with st.container(horizontal=True, width="content", gap=None, key="aw_modes"):
            for mode, icon in MODES.items():
                if st.button(icon, key=f"aw_mode_{mode}", type="primary" if mode == now else "tertiary",
                             help=f"{mode} glass (kept in this browser)"):
                    st.session_state["_aw_switch_mode"] = mode      # apply() stores it and reloads on the next run
                    st.rerun()


def section(label: str) -> None:
    """A small upper-case section label in the sidebar ("VIDEO", "DETECTOR")."""
    st.html(f'<div class="aw-section">{html.escape(label)}</div>')


def footer() -> None:
    """The last line of the sidebar: where the analysis runs (any computer with the project installed)."""
    st.html('<div class="aw-foot">Private: videos are analysed on the computer that runs AquaWatch, with no cloud '
            'service, and this page opens only on that computer.</div>')


def overline(text: str) -> None:
    """A small blue upper-case line above a page title ("MULTI-CAMERA SITE")."""
    st.html(f'<div class="aw-overline">{html.escape(text)}</div>')


def empty_state(icon: str, title: str) -> None:
    """A centred icon and title for an empty panel (the text under it is drawn by the caller with st.caption)."""
    st.html(f'<div class="aw-empty"><div style="font-size:2rem">{html.escape(icon)}</div>'
            f'<div class="aw-empty-title">{html.escape(title)}</div></div>')


def glass(key: str, **kwargs):
    """A glass panel: st.container(key="glass_<key>") (the theme above styles every key that starts with glass)."""
    return st.container(key=f"glass_{key}", **kwargs)
