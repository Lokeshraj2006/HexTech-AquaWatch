# Example result: `pool_synthetic/`

A finished AquaWatch analysis you can open without running anything: in the dashboard pick
**Previous results -> `data/examples/pool_synthetic`**, or open `pool_synthetic/report.html` in a browser (one file,
images embedded).

## What it is

The drawn test story (a scene script, not real footage): a 40 s clip with 5 people. A lap swimmer, a swimmer resting at
the wall, a diver and a lifeguard on the deck must stay normal; child #3 swims, slows down, turns upright, presses the
arms down, and then the head goes under.

**Important:** the boxes and pose points of this story are written by the scene script (what a perfect detector would
give), not found by YOLO. So this example shows the behaviour rules, the alerts and the outputs. It does **not** show
how well the detector works on real footage.

Made with the `pool` preset on 2026-10-07 (`meta.generated` in `events.json`), before the rule change of 2026-10-11.
The project's local test suite still checks this story with today's rules.

## The result (from `summary.txt` and `events.json`)

- Swimmer #3: **High-risk aquatic distress**: the pattern started at 00:17.0 and the alert came at 00:22.0 in the Deep
  End, risk 97%. Evidence: vertical posture, little or no forward progress, repeated distress-like arm movement,
  unstable head, was swimming normally just before.
- Swimmer #3: **Possible submersion** at 00:29.8 (lost from view after the distress).
- Both are linked into one incident: "Distress in the water, then went under". Nobody else was flagged.
- Ground truth in `labels.csv`: distress 16.0-28.0 s, head goes under 28.0-30.0 s.

## Files

| File | What it is |
|---|---|
| `report.html` | the full evidence report (one file) |
| `summary.txt` | the short summary |
| `events.json` | every event with times, evidence, confidence, severity, incidents, and the settings used |
| `stage3_events.json` | stage 3 output: alerts and the timeline per swimmer |
| `pool_dashboard.json` | what the dashboard shows: people, risk curves, timelines, alerts, thresholds |
| `announcements.json`, `alerts/alert_e1.wav`, `alerts/alert_e2.wav` | the spoken alerts (text and sound) |
| `snapshots/e1.jpg`, `snapshots/e2.jpg`, `plots/e1.png`, `plots/e2.png` | evidence picture and risk plot for each alert |
| `behaviour_plots/person_3_signals.png` | the behaviour signals of swimmer #3 over time |
| `timeline.png` | who was visible when, with the alerts |
| `highlights.mp4` | only the incidents, captioned |
| `labels.csv` | the ground truth of the story |
| `ask_index.sqlite3`, `references.sqlite3` | written by the chat engine (`ps_conversation.folder_conversation`) when it answers about this folder: the event index and the saved places |

There is no `annotated.mp4` and no original clip here (kept small), so the dashboard shows the results without the replay video.

## How it was made

A scene script drew the clip and wrote its detections, pose points and labels; then
`python aquawatch/run.py --scenario pool --reuse` ran only the rules on them (`--reuse` skips the detector). This
folder keeps the result of that run (made 2026-10-07) without the videos. A new run uses today's rules, so its times
can differ slightly.
