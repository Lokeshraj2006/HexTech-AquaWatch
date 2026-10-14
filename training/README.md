# Training kit: swimmer body-state detector

## What we train
We fine-tune **YOLO11n** to find every swimmer and output their **body state** in each frame. It is **not** a drowning classifier. The distress decision is made over time by `distress.py`.

| id | class | meaning |
|---|---|---|
| 0 | `swimmer_horizontal` | body flat, swimming |
| 1 | `swimmer_vertical` | body upright in the water. This covers treading water **and** distress; the model must not decide which |
| 2 | `head_only` | only the head is above the water |
| 3 | `underwater` | body visible below the surface |
| 4 | `out_of_water` | on the deck (the pipeline ignores these) |

**Model output** (after ByteTrack): `{"person_id", "timestamp", "bbox", "conf", "class", "state"}`.

## Steps (what we actually did)
1. **Download.** `python training/get_datasets.py` downloads the 3 Roboflow Universe datasets into `datasets/raw/`. Set your key first in your own terminal: `$env:ROBOFLOW_API_KEY = "..."`. The script uses IPv4 only and retries, because on some networks Roboflow's IPv6 route times out.
2. **Merge.** `python training/build_dataset.py` reads `training/pool_datasets.yaml` and builds `datasets/swimmers/`. It does the following:
   - **Maps labels** after looking at sample boxes of every class:
     - `Swimming` → horizontal
     - `Drowning`, `drowning` and `treading_water` → vertical (treading and distress look the same in one frame; stage 3 tells them apart over time)
     - `Person out of water` → out_of_water
   - **Removes duplicates, checked on the pixels.** `poolsafety` turned out to be a re-upload of `university` (40 of 40 sampled frames identical), so it is disabled. Of `treading`, 729 copies were dropped and 407 same-name-but-different images were kept.
   - **Splits by video, across all sources.** A video is in one split only, so the test score cannot be inflated by near-identical frames.
   - **Writes** `data.yaml`, `manifest.csv` and `CREDITS.md`.

   Current result: **13,081 images**. Train is 10,408 (573 videos), val is 1,310 (131 videos), test is 1,363 (100 videos). There are no `head_only` or `underwater` boxes yet; label frames from your staged clips to add them (class ids stay the same).
3. **Quick check on a laptop CPU** (about 5-10 min, not accurate, proves the pipeline works):
   ```bash
   python training/train_swimmer.py --data datasets/swimmers/data.yaml --device cpu --quick
   ```
4. **Real training on Colab with a T4.** Use `training/colab_run.ipynb` (section 4). Put the key in Colab **Secrets** as `ROBOFLOW_API_KEY`. The section downloads, merges, trains (`--epochs 60`, about 30-40 min), shows the curves and downloads `best.pt` and `eval.json`.
5. **Plug it in.** Put `best.pt` at `training/runs/swimmer/weights/best.pt` and use the **`pool_trained`** preset
   (`scenarios/pool_trained.yaml`), which already has these lines:
   ```yaml
   model:
     weights: training/runs/swimmer/weights/best.pt
     classes: [0, 1, 2, 3, 4]
   pool:
     state_classes: {0: swimmer_horizontal, 1: swimmer_vertical, 2: head_only, 3: underwater, 4: out_of_water}
   ```
   `python aquawatch/run.py --video pool.mp4 --zones pool_zones.json --scenario pool_trained`. Keep `pool` (stock detector) for
   the synthetic story: its cached detections belong to the stock model. After a retrain, `--reuse` notices the new
   `best.pt` (its SHA-1 is stored in `detections.json`) and runs the detector again.
6. **Report honestly:**
   - mAP per class on the **test** split (`eval.json`).
   - Where it fails (glare, crowding, top-down drone frames in the data).
   - The full pipeline's alerts and false alarms on your **staged** clips.

The datasets are CC BY 4.0. Copy `datasets/swimmers/CREDITS.md` into the README's declared resources.

## Tips
- Keep **vertical flip off**: an upside-down swimmer would teach the wrong posture. The script already sets this.
- **Class balance matters.** `head_only` and `underwater` are usually rare. Add frames from your own staged clips, labelled with any labelling tool that exports YOLO format.
- **Use the same camera height and angle** as the pool you will demo in. Domain match beats dataset size.
