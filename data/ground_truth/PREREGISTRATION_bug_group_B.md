# Pre-registered check: bug hunt group B (pool / drowning-rule findings)
(written 2026-10-12, BEFORE any of these fixes is coded or run)

Source: `docs/BUG_HUNT_2026-10-11.md`, section "pool" (9 findings, all CONFIRMED). Each is a place where the code does
not do what its own documented rule says. Fixing them changes which alerts fire, so they are applied one at a time
and measured.

## Order (most severe first) and expected direction
| Step | Finding | Expected effect on the labelled runs |
|---|---|---|
| B1 | treading-water cap skipped when arms 'irregular' or head 'unknown' (distress.py:60) | fewer false Level 1 warnings |
| B2 | box-top bobbing counted twice without pose (arms + head) | fewer false warnings |
| B3 | DISTRESS risk_score falls below announce.min_risk after the alarm (keep the max) | none on alerts (only the score / speech) |
| B4 | 'passive drowning' fires on a bobbing head (needs head unknown / submerged) | fewer false warnings |
| B5 | Rule C / A findings dropped for a swimmer without stage-2 samples | maybe more Rule C alerts (true or false) |
| B6 | Rule C checks 'spot stayed empty' on filtered tracks (use raw detections) | Rule C alerts may change either way |
| B7 | the last 0-0.5 s of a track is never sampled | a few more samples; maybe more warnings |
| B8 | Rule B counts neighbour samples, not swimmers (Rule B is OFF) | none (latent) |
| B9 | hold_s counted across gaps with no valid sample (reset the pending run) | fewer false DISTRESS alarms |

## Measure
After each step: `outputs/gt_eval/run_vanish_waves.py B<n>` (same 8 labelled runs, saved detections) and
`tools/eval_alerts.py --runs outputs/gt_eval --tag B<n>` (corrected scoring). Reference: tag G4 = 3/6 victims caught,
13 false alerts. Also the demo single clip and the 4-camera site at the end.

## Acceptance (decided now)
- A step is KEPT if, with it and the kept steps before it, victims caught stay >= 3/6 AND false alerts stay <= 13.
- A step that breaks either limit is NOT kept: the code goes back, and the result is reported. (A correctness fix that
  costs a catch is a real finding about the rules, not something to tune away.)
- Steps with no effect on the labelled runs (latent bugs) are kept: they make the code do what it says.
- No threshold is changed in this group.

## Amendment B1b (written 2026-10-12 AFTER steps B1-B9 were measured, BEFORE B1b is coded)
What happened: all nine steps passed the acceptance above (final: 3/6 caught, 13 false; B1 -2 false, B5 +2 false), but
the full test suite then failed: in the drawn test story (tests/synthetic_pool.py) the child in DISTRESS who goes under
no longer gets "possible submersion". Cause, traced: B1. In the child's last 0.5 s the samples read upright, arms
'irregular', head 'stable' = treading by the corrected rule, risk capped to 0.45 < release_risk 0.5, so the DISTRESS
alarm ends just before the child vanishes and the submersion check never runs. The acceptance above looked only at the
labelled real runs, which never reach Level 3, so it could not see this: a gap in this pre-registration.
Amendment (designed after seeing this failure, said openly): the treading cap may stop an alarm from STARTING, but
it never ENDS an alarm already raised: while in DISTRESS, the release test uses the risk without the treading cap.
Acceptance: labelled runs still >= 3/6 caught and <= 13 false alerts; the whole test suite passes, including the
drawn story's submersion. If not, B1 is not kept.
