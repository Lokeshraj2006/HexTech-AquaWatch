# Pre-registered check: camera moving / top-down view warnings (REAL_WORLD_CHECK G7)
(written 2026-10-12, BEFORE the code and BEFORE measuring any clip)

Why (REAL_WORLD_CHECK C): a handheld phone (14.42.58 = 15.57.23) breaks the movement signals and a drone / ceiling
view (pexels_32723756, pexels_4926273) breaks the posture-from-box-shape signals, but AquaWatch analyses them
silently as if they were fixed side-view CCTV.

## The two warnings (orange "results less reliable", not the red feed problems; they never change an alert)
- **Camera moving:** on the feed check's 64 px grey samples (2 per second), the whole-picture shift between two
  samples (OpenCV phase correlation) is at least `move_frac` = 0.02 of the picture width (= 4 % of the width per
  second) for at least `problem_s` = 2 s.
- **Top-down view:** from the detections: the median box height in the bottom third of the picture divided by the
  median box height in the top third is below `top_down_ratio` = 1.3 (in a side view far people at the top look
  smaller), with at least `view_min_boxes` = 20 boxes in each third; fewer boxes = no conclusion (no warning).

## Expected (from REAL_WORLD_CHECK C, decided now)
| Clip | Moving? | Top-down? |
|---|---|---|
| 14.42.58, 15.57.23 (handheld phone) | YES | not expected |
| pexels_32723756 (drone, moving) | YES | YES if enough boxes (few expected: no conclusion is acceptable) |
| pexels_4926273 (top-down) | unknown | YES |
| 2.36.05, 14.26.16, 14.42.46 (fixed wide CCTV) and cam01-04 (crops of 2.36.05) | NO | NO |
| 14.32.56, 15.59.14, 07.21.29, 07.21.30, pexels_8685964 | unknown | unknown |

## Acceptance
- A "NO" clip that gets a warning, or a "YES" clip that does not, is reported as a miss. Thresholds are not tuned
  after measuring; a wrong threshold is reported and fixed only under a new pre-registration.
- Clips marked unknown are reported as measured, checked by eye where possible.
