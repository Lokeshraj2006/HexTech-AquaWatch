# Pre-registered check: "vanished in the water" and "wave-relative head" (written 2026-10-11, BEFORE the code)

Why (the user, after watching the 4-camera demo): the drowning victim gets a box and an ID in some cameras but never
a risk, and people calmly floating in tubes get warnings.
What we found before writing this (outputs/virtual_cctv, data/ground_truth):
- The labelled victim (V3, the person in the blue tube on the right) slips through the tube at about 6.5-7.5 s and is
  then UNDER the water (a dark shape, 9-14 px tall) until the footage ends. The detector sees her only in flickers
  (CAM03: 59 of about 600 frames), so the behaviour rules never get the 2 s of steady view they need: no risk at all.
  CAM01 and CAM02 do not contain her (crop geometry).
- In CAM03 her track while she sits in the tube ends at 6.51 s, when she slips under.
- The 5 warnings left on the demo site all rest on "vertical posture + little progress + unstable head"; in a wave pool
  the waves move every head.

## Rule A: vanished in the water (new Level 1 reason, "lost from view in the water")
A swimmer who was followed for at least `vanish_min_seen_s` = 3.0 s and then disappears is flagged when ALL hold:
the video goes on for at least `vanish_lost_s` = 3.0 s after the last sighting; the last box is at least 1 box height
away from every frame edge (not walking out of the picture); no other track starts within 2 box heights of the
last position during those 3 s (stitch.py already joined the plausible continuations); not at a wall / in a zone
where standing is normal. One sentence: "someone we were watching vanished in open water and did not come back".

## Rule B: wave-relative head
"Unstable head" from bobbing counts only when the swimmer bobs more than the swimmers around them:
own bobbing minus the median bobbing of the other swimmers within 4 box heights at the same moment must exceed
`head_unstable` (0.06). With fewer than 2 such neighbours, nothing changes. A head that is low above the shoulders
still counts as before. One sentence: "a wave lifts everybody; only bobbing more than the neighbours is a sign".

## Measures (tools/eval_alerts.py, written before; same runs and detections as alerts_after.json)
Victims caught and false alerts on V1-V3 (pool_hd, pool_trained) and N1-N2 (pool_trained), as in
PREREGISTRATION_alerts.md. Also reported (not part of the decision): the 4 virtual cameras, with the V3 positions
shifted into each crop.

## Acceptance (each rule judged on its own, then both together)
- Rule A is kept only if victims caught go UP (from 0) and false alerts in total stay at or below 15 (today 7, before
  the 2026-10-11 fix 16).
- Rule B is kept only if false alerts go DOWN and victims caught do not go down.
- No threshold is tuned after seeing the results. If a rule fails, it stays in the code switched off, and the report
  says so.

---

## Results (added after the runs, 2026-10-11; nothing above was changed)

Runs: `outputs/gt_eval/*_{base,A,B,AB}` (`outputs/gt_eval/run_vanish_waves.py`), scored with `tools/eval_alerts.py`.

| Variant | Victims caught (pool_hd + pool_trained) | False alerts (8 runs) | Verdict |
|---|---|---|---|
| base (current rules) | 0/3 + 0/3 | 2 + 5 = 7 | - |
| A: vanished in the water | 1/3 + 3/3 (V1 +16.5 s, V2 +9.6/+10.3 s, V3 +1.3 s) | 22 + 23 = 45 | **fails** (limit 15) |
| B: wave-relative head | 0/3 + 0/3 | 7 | **fails** (no fewer false alerts) |
| A + B | 1/3 + 3/3 | 45 | fails |

- Rule A finds the victims, but in crowded pools people vanish behind others and in waves all the time: about 6
  false alerts per clip. With that many alerts some of the "catches" may also be luck. Kept in the code, off.
- Rule B: the first version grouped samples by exact time stamp, but every swimmer is sampled on its own clock, so
  it found no neighbours and changed nothing. Fixed to "within half a sample step" (the rule as written; no threshold
  changed) and run again: 2-6 heads per run become "stable", no alert goes away. On these clips the warned heads
  really bob more than their neighbours, or sit low above the shoulders. Kept in the code, off.
- Information only, the 4 virtual cameras (V3 positions shifted into each crop,
  `outputs/gt_eval/site_vanish_check.py`): base catches the victim on no camera, 5 false alerts; Rule A catches her
  on CAM03 (its only alert there, 3 s after she slips under) but has 17 false alerts over the 4 cameras.

- Scoring correction (2026-10-11, see PREREGISTRATION_same_spot_zones.md): alerts about someone who vanished are
  now scored at their alert time and the place where that person was last seen. Rule A becomes 5/6 victims, 44
  false alerts: still fails (limit 15).
