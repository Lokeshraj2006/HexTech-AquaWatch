# Pre-registered check: bug hunt group C (tracking findings)
(written 2026-10-12, BEFORE any of these fixes is coded or run)

Source: `docs/BUG_HUNT_2026-10-11.md`, section "tracking" (7 findings + 1 found by the skeptic, all CONFIRMED).
Reference: after group B (tag B1b) = 3/6 victims caught, 13 false alerts on the 8 labelled runs (corrected scoring).

## Phase 1: fixes that act on the SAVED detections (one at a time, --reuse, ~1 min each)
| Step | Finding | Expected effect |
|---|---|---|
| C1 | min_box_h_px is measured in processing pixels: take it in ORIGINAL video pixels | more boxes kept for wide / high-res views; tracks longer; more samples (could add warnings) |
| C2 | stitch allowance grows without limit: cap it at stitch.max_jump_bh = 3 body sizes | fewer wrong joins (cam03 #15 should keep its own ID); Rule C may change |
| C3 | stitch size ratio by box height; use the body scale (max side) like features | a swimmer turning flat -> upright keeps the ID |
| C4 | gaps for stage 3 from the filtered track: take them from all of that ID's boxes | latent here (no DISTRESS episodes in the labelled runs) |

## Phase 2: fixes that change the detector output (fresh runs, ~1 h of CPU, one at a time)
| Step | Finding | Expected effect |
|---|---|---|
| C5 | ByteTrack never gets low-confidence boxes (conf passed = its high threshold): pass conf = track_low_thresh; for bytetrack_custom.yaml raise track_high / new_track to 0.30 (the old model conf) so no new junk IDs start | fewer ID switches for half-hidden swimmers |
| C6 | --reuse ignores conf / iou / imgsz / tracker yaml, accepts a longer cache, and (command line) a different video size | none on results; old caches without these fields are no longer reused (by design) |

Confound, known now: some saved detections were made on the old GPU laptop with the .pt model; a fresh run here uses
the OpenVINO CPU export, so a fresh run can differ for reasons other than C5. Phase 2 is measured as "fresh runs with
C1-C5"; if it fails the acceptance, fresh runs WITHOUT C5 are made too (another ~1 h, asked first) so the cause is
known before anything is decided.

## Measure
Phase 1: `outputs/gt_eval/run_vanish_waves.py C<n>` + `tools/eval_alerts.py --runs outputs/gt_eval --tag C<n>`.
Phase 2: the same 8 labelled runs WITHOUT saved detections (fresh detector + pose), tag C5; then the 4 demo cameras.
After every step also the whole test suite (the drawn stories guard Level 2 / Level 3, which the labelled runs never
reach: lesson of group B).

## Acceptance (decided now)
- A step is KEPT if, with the kept steps before it, victims caught stay >= 3/6 AND false alerts stay <= 13 AND the
  whole test suite passes. Otherwise it is reverted and reported.
- Latent steps (no effect on the labelled runs) are kept.
- No rule threshold is changed; the only new setting is stitch.max_jump_bh = 3 (from the bug hunt, chosen before
  measuring) and the bytetrack_custom.yaml thresholds named above.

## Results so far (added 2026-10-12, after measuring)
- Phase 1: C1-C4 all kept: 3/6 caught, 13 false alerts after each (C2 moved false alerts between runs: V1 +2,
  V3 posture model -2). Full test suite passed after each (C1 needed one test updated to the new meaning).
- C5 (fresh runs with C1-C5, 50 min): **2/6 caught, 24 false alerts: FAILED** (V3 posture-model catch lost; false
  alerts V1 Wide CCTV 2 -> 9, V3 Wide CCTV 0 -> 4, N1 1 -> 3). As planned above, C5 is reverted and the 8 runs are
  made fresh WITHOUT C5 (tag C5base) to know whether C5 or 'a fresh CPU run' causes the change. Decision rule for
  C5 (written now, before C5base is measured): C5 is kept only if C5 is no worse than C5base on both counts
  (caught >= C5base, false <= C5base) AND within the original limits; otherwise it stays reverted.- Correction (2026-10-12, checked after the user pointed out that this laptop has no GPU): the confound above was
  overstated. Of the 8 saved labelled detection files, 7 were made on THIS laptop's CPU (OpenVINO); only
  V3_pool_hd_after was made with a GPU (device 0, yolo11s.pt, the team's earlier laptop). So "GPU vs CPU" can explain
  at most one run, and V3 Wide CCTV still caught the victim with C5. What remains for C5base to separate: C5 itself
  vs older code / settings at the time the saved files were made (and the V3 video trimmed on 2026-10-11).

