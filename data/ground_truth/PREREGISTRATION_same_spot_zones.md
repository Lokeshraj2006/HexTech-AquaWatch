# Pre-registered check: "back at the same spot" (Rule C) and pool-edge zones for the demo site
(written 2026-10-11, BEFORE the code and the runs)

Why (the user, watching the 4-camera demo again): the victim behaves like a drowning person but gets no alert, and
her ID changes (CAM03 #4 then #15, CAM04 #5 then #120); a person resting at the wall in CAM01 and swimmer #15 in CAM04
get "swimmer may need help".
What we found before writing this:
- The victim is boxed while she sits in the tube (CAM03 #4 until 6.51 s, CAM04 #5 until 4.84 s), then only when her
  head breaks the surface: CAM03 #15 at ~15 s and 19-20 s, CAM04 #120 at 15-20 s, at the SAME spot each time
  (within about 1 box height). The gaps (8.5 s, 10 s) are longer than stitch.max_gap_s (5 s), so she becomes a new ID
  and no piece is seen steadily enough to be judged.
- CAM01 #10 / #26 and CAM04 #15 are people standing in the water at the left wall, by the ladder and the steps. The
  demo site has no zones, so the rules do not know where the wall is (resting at the wall is normal: risk x 0.3).
- HONEST CAVEAT: Rule C was designed after looking at this victim (V3). V1 is the same recording in another framing,
  so only V2 (the toddler) and the calm clips N1/N2 are an independent test.

## Rule C: back at the same spot (new Level 1 reason)
A swimmer followed for at least vanish_min_seen_s (3 s) disappears with the last box at least vanish_edge_bh box
heights from every picture edge, not at a wall / where standing is normal; for at least `same_spot_min_s` = 5 s
nobody is seen within `same_spot_bh` = 1.5 box heights of that spot; then, within `same_spot_max_s` = 20 s after the
disappearance, somebody (any track) is seen there again. The alert is raised at that moment: "out of sight for N s,
then seen again at the same spot (may have been under the water)". One sentence: "someone vanished in open water and
something came back up in the very same place many seconds later".

## Pool-edge zones for the demo site (site configuration, not a rule)
`data/virtual_cctv/zones_source.json` (source pixels, drawn by eye on a gridded frame BEFORE any run): a "Pool edge"
band along the left wall and steps (30 px wide at the far end to 70 px near the camera, plus the deck) and along
the right wall (30 px). No "water" outline (a wrong one would hide real swimmers). Copied into each camera's
pixels (`data/virtual_cctv/zones/camXX_zones.json`) and set in cameras.yaml. The labelled victim (~(770, 190)) is
about 60 px from the right wall: outside the band.

## Measures
- Rule C: tools/eval_alerts.py on the same 8 labelled runs as PREREGISTRATION_vanish_waves.md (no zones there).
- Zones: the 4 demo cameras, false alerts and the V3 victim (positions shifted into each crop), with and without zones.

## Acceptance
- Rule C is switched on only if victims caught go UP (from 0) and false alerts in total stay at or below 15 (today 7).
- The zones are kept for the demo site if its false alerts go down and the victim is not hidden by them (her samples,
  if any, are not marked near_wall). Zones never change the labelled evaluation runs.
- No threshold is tuned after seeing the results. A rule that fails stays in the code, switched off, and is reported.

---

## Results (added after the runs, 2026-10-11; nothing above was changed)

**Rule C on the 8 labelled runs** (`outputs/gt_eval/*_C`, `outputs/gt_eval/run_vanish_waves.py C`, tools/eval_alerts.py):

| | Victims caught (pool_hd + pool_trained) | False alerts |
|---|---|---|
| current rules | 0/3 + 0/3 | 2 + 5 = 7 |
| Rule C | 0/3 + **1/3** (V3, +1.3 s, by a Rule C alert) | 5 + 10 = **15** |

- Passes the bar as written (victims up, false alerts at or below 15) -> switched ON (`pool.same_spot_enabled: true`).
- Read it honestly: the one catch is V3, the clip Rule C was designed from; V1 (same recording, other framing) and
  V2 (independent) are not caught. All 8 extra false alerts come from Rule C (V1: 6, V3: 2); the calm clips N1 and N2
  got none. Promising, not proven.

**Demo site, 4 cameras** (`outputs/gt_eval/site_same_spot_zones_check.py`, V3 positions shifted into each crop):

| | Victim caught | False alerts |
|---|---|---|
| current rules | no camera | 5 |
| current rules + pool-edge zones | no camera | 1 |
| Rule C | CAM03 | 6 |
| Rule C + pool-edge zones | CAM03 | 2 |

- Zones: false alerts 5 -> 1 and the victim is not hidden (Rule C still catches her on CAM03 with zones) -> kept
  for the demo site (cameras.yaml) and for the single clip of the same pool
  (`samples/uploads/WhatsApp_Video_2026-10-07_at_2.36.05_PM_zones.json`). The labelled runs above never used zones.

### Scoring correction (found after the runs above, 2026-10-11; reported, not hidden)
tools/eval_alerts.py took an alert's time from its snapshot. For an alert about someone who VANISHED (Rules A and C)
the snapshot is pinned to their last sighting, so a warning raised at 15.0 s about the victim who slipped under was
scored at 4.4 s (before her distress began) and counted as a false alert, although the person flagged was the victim.
The definition above says "at the alert time"; the scorer now uses the alert time and the place where that person was
last seen (compared with where the victim was then). Base and Rule B scores do not change.

| 8 labelled runs | Victims caught (first scoring) | False (first) | Victims caught (corrected) | False (corrected) |
|---|---|---|---|---|
| current rules | 0/6 | 7 | 0/6 | 7 |
| Rule A | 4/6 | 45 | 5/6 | 44 |
| Rule C | 1/6 | 15 | **3/6** (V1 pool_hd, V3 both) | **13** |

The decisions do not change (Rule A fails at 44 > 15; Rule C passes either way). Rule C's catches are on V1 and V3,
which are the same recording; V2 (independent) is still missed.

Demo, after switching Rule C and the zones on (corrected scoring): the single clip catches the victim (+8.5 s,
0 false alerts); the 4 cameras catch her on CAM03 (+8.4 s) and CAM04 (+8.5 s) with 1 false alert in total (CAM02).
