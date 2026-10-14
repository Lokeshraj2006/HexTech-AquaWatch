# Pre-registered check for the distress rules (written 2026-10-11, BEFORE any rule change)

Why: a calm person standing in shallow water was warned ("upright + not moving 4 s") in
WhatsApp_Video_2026-10-09_at_07.21.30 (yellow shirt, Swimmer #3, 6-8 s). Any fix must not cost us a real victim.

## Ground truth used
`data/ground_truth/real_clips_ground_truth.json` (labels from an earlier session's frame-by-frame inspection,
confidence "high"; not an independent human annotation):
- V1 `samples/uploads/WhatsApp_Video_2026-10-07_at_14.26.16.mp4`: victim distress 6.5-23.5 s (positions listed);
  hard negatives: a front-crawl swimmer 13-17 s, an underwater swimmer 25.5-34.5 s.
- V2 `samples/uploads/WhatsApp_Video_2026-10-07_at_14.32.56.mp4`: toddler distress 5.0-19.3 s (7.7-17.2 s is a zoomed
  replay; positions in that part are in zoomed pixels).
- V3 `samples/uploads/WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4` (= 14.36.05, same footage, 848x480): victim
  distress 6.5-21.6 s; positions = the json's V3 positions.
- N1 `WhatsApp_Video_2026-10-09_at_07.21.30.mp4` and N2 `..._07.21.29.mp4`: people standing in shallow water; nobody
  is in distress (assumption stated by the team's own description of the clip; the red-shirt splasher is acting).

## Measures (per clip, per preset; computed by tools/eval_alerts.py, code written before the rule change)
- **Victim caught** = an alert (any level) with its time inside [distress_t0, distress_t1 + 2 s] whose track box
  centre, at the alert time, is within max(2 x victim box height, 30 px) of the victim's labelled position (linear
  interpolation between labelled times; video pixels).
- **Time to first alert** on the victim (alert time - distress_t0).
- **False alerts** = every other alert (any track that is not the victim at that time), on V1-V3 and N1-N2.

## Acceptance of a rule change
1. Victims caught must not go down (on any preset).
2. False alerts must go down in total.
3. The rule must be explainable in one sentence (no thresholds tuned to these 5 clips one by one).
If a change cannot meet 1 and 2, it is not adopted, and the report says so.
