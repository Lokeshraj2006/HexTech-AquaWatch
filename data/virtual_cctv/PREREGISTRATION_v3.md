# Pre-registered decision rules: open-vocabulary detection + quantization (v3)

Written on 2026-10-10, **before** any YOLO-World / YOLOE grounding code, quantized model or v3 result existed.
The only v3 facts known at this time: a one-frame smoke test (both detectors load and run at about 350 ms per frame,
640 px, 2 CPU threads; neither called the jumping lifeguard "lifeguard"). These rules are not changed after results
are seen; if a rule turns out to be unworkable, the change and the reason are written in the results report (now docs/RESULTS.md) section N.

## Data
- Query sets: `openvocab_queries.yaml` (set 1: 8 + 3 negatives), `openvocab_queries_v2.yaml` (set 2: 6 + 3) and
  `openvocab_queries_v3.yaml` (set 3: 5 + 3). Sets 1-2 were used to pick the CLIP frame baseline (section M) but never
  to design the grounding methods; set 3 is new. **Primary metric: all 19 positives and 9 negatives pooled.**
- Hit rule (unchanged): a returned hit on an expected camera, time inside the window +- 1 s.
- Labels are AI-drafted from frames (not an independent human annotation): this is stated next to every number.

## Fixed defaults (chosen before results, not tuned on any query set)
- Grounding ("retrieve, then detect"): stage 1 = the CLIP frame baseline, K = 12 candidate frames in scope; stage 2 =
  the open-vocabulary detector run on those frames with the query text as its only class, 640 px, confidence
  threshold 0.25 (the Ultralytics default). Best box confidence < 0.25 on all K frames -> "no confident match".
- Fixed-vocabulary index (every 0.5 s, every camera, 640 px, conf 0.25), used only when the whole query is one
  vocabulary term (or its plural): person, child, lifeguard, swimmer, life jacket, inflatable ring, rescue tube,
  swim cap, goggles, umbrella, ladder, rope, buoy, splash, towel, chair, bag, backpack, hat, sunglasses, ball, phone,
  bottle, dog, car, bicycle.
- Development checks while building may only use query texts that are in none of the three query sets.

## R1. Which method Ask the Video uses
1. Candidates: `baseline_frame` (current) and every new method (YOLO-World / YOLOE grounding, fixed-vocabulary index).
2. Rank by pooled hit@1. A new method replaces the baseline only if
   (a) its pooled hit@1 is higher, or
   (b) its pooled hit@1 is at most 1 lower AND its hit@5 is not lower AND it rejects at least 6 of the 9 negatives
       (the baseline rejects 0: it cannot say "not there").
3. Ties between new methods: higher hit@5, then more negatives rejected, then lower median latency.

## R2. YOLO-World vs YOLOE
- The same rule as R1 between the two grounding methods (same retrieval stage, same K, same thresholds).
- A colour-order bug check runs first (the smoke test image of YOLOE looked red/blue swapped); results are only
  compared after both backends are confirmed to receive the frame in the colour order they expect.

## R3. INT8 person detectors (yolo11n, yolo11s, our best.pt)
- Reference: the same model in FP32 (PyTorch). Data: 100 frames sampled evenly from the 4 virtual cameras + the
  single-camera sample clip (the frames are not used for INT8 calibration; calibration uses other frames).
- Agreement: boxes matched one-to-one at IoU >= 0.5 (conf >= 0.25, same class); F1 against FP32.
- INT8 is adopted for a model only if F1 >= 0.98. Otherwise the FP32 export (ONNX / OpenVINO, numerically the same
  model) is used if it is faster; else PyTorch stays.
- For best.pt the Colab test split is not on this laptop: no mAP claim, only agreement. If the dataset is supplied,
  mAP50 of FP32 vs INT8 on the test split is measured and INT8 needs a drop <= 1.0 mAP50 point.

## R4. INT8 CLIP (dynamic quantization) and INT8 stored embeddings
- Adopted only if, on the pooled sets, hit@1 and hit@5 of the chat method each lose at most 1 query vs FP32 / FP16.

## R5. Fixed-vocabulary detector exported to INT8
- Same agreement rule as R3 (F1 >= 0.98 vs its FP32 PyTorch run, same frames).

## Reported for every variant
Pooled hit@1, hit@5, negatives rejected, median and max latency per query, model file size, and (detectors) frames
per second on this laptop (AMD Ryzen 5 7520U, 2 threads, below-normal priority).

## Addendum A (2026-10-10, before the final R1/R2/R3 runs; after the builders' development checks)
Written after the reviewers flagged two ambiguous readings. In each case the STRICTER reading is taken.
- **R4 "lose at most 1 query"** is counted per query: a variant fails when more than 1 individual query that the
  reference found is no longer found (at hit@1, or at hit@5), whatever it gains elsewhere. Net counts are reported too.
  Under this reading LAION clip_int8 (2 individual losses at hit@1) FAILS R4; it also changes the ranking a lot
  (frame-score Spearman 0.65 vs fp32 on development texts), so it is not used.
- **CLIP model.** Before any result, the user was told: "if OpenAI CLIP turns out as good for search as the LAION model
  we installed, I'd use one CLIP for everything" (YOLO-World needs the OpenAI CLIP anyway; the LAION copy is 578 MB
  more on disk). The development-free R4 runs give OpenAI 12/19 at hit@1 and 18/19 at hit@5 against LAION 11/14, so
  OpenAI is at least as good: the chat switches to it, and R1 is run with the OpenAI index for every method (baseline
  included), so the comparison stays like for like. LAION numbers stay in the report.
- **R3 "INT8 only if F1 >= 0.98"** is a necessary condition, not a sufficient one: INT8 must also be faster than the
  PyTorch reference, otherwise it brings no benefit. Between the two FP32 exports (identical boxes), the faster median
  wins; if they are within 10 % of each other, OpenVINO FP32 is taken (its thread cap was verified by read-back).
- The INT8 timing warm-up now includes one frame of every input shape (4:3 and 16:9 videos), because a dynamic
  export pays a one-off cost for a new shape; this changes timing only, never boxes or thresholds.
