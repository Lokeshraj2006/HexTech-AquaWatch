# Pre-registered check: Rule C as its own urgent alert type, alerts that say where (REAL_WORLD_CHECK G4)
(written 2026-10-12, BEFORE the code changes and the runs)

Why (REAL_WORLD_CHECK.md F3 and F4): the only alert that catches real victims (Rule C, "out of sight, then seen
again at the same spot") is shown as the softest Level 1 "please check" with a soft chime, mixed with all the other
Level 1 warnings; and 100 % of real alerts say "at the Pool" plus an internal track number, with no camera name.

## What changes (presentation only: no rule decides differently)
1. A Rule C alert gets its own urgent type, "POSSIBLY UNDER WATER", ranked above Level 1 and below the Level 2 alarm:
   its own title, colour, KPI tile, status-board state, the urgent alarm tone and an urgent spoken line.
   The stored event stays `early_warning` with evidence `back_at_same_spot` (behaviour `aquatic_warning`), so the
   evaluation, the SQLite index and old results read the same; the type is derived from that evidence.
2. The spot where the swimmer vanished is marked on the annotated video (and snapshot) when the alert fires.
3. Every alert and spoken line says where: camera name (when known: multi-camera site, or typed by the user), the zone
   ("Pool" / zone name) and the part of the picture (e.g. "top left of the picture").
4. No "percent" in speech (the risk is a weighted sum, not a probability).

## Measure
tools/eval_alerts.py on the 8 labelled runs (`outputs/gt_eval`), as in section Q, with saved detections (--reuse):
run `outputs/gt_eval/run_vanish_waves.py G4` then `tools/eval_alerts.py --runs outputs/gt_eval --tag G4`.

## Acceptance (decided now)
- Victims caught, false alerts and every alert time must be IDENTICAL to tag C (today: 3/6 caught, 13 false alerts,
  corrected scoring). Any difference means the change touched a rule by mistake: it is a bug, fixed before keeping G4.
- The demo single clip and the 4-camera site keep the same alerts (single clip: Swimmer #5 at 00:15; site: CAM02,
  CAM03, CAM04), now with the Rule C ones shown as POSSIBLY UNDER WATER where they come from Rule C.
- No threshold is changed.
