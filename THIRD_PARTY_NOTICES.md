# Third-party notices

AquaWatch is built on the open-source software, model weights and datasets below. Licences were read from the
installed packages' own metadata in `.venv` on 2026-10-12 (versions as installed). This file is a summary, not legal
advice: each licence text is the binding one.

## Important for any use outside this project

- **Ultralytics (AGPL-3.0).** YOLO detection, YOLO pose, ByteTrack (via Ultralytics), YOLO-World and YOLOE, and every
  weight file made with Ultralytics (including our own `best.pt` and `models/pool_person_s800`), fall under AGPL-3.0.
  Anyone who runs AquaWatch as a network service or distributes it must offer the complete source code under
  AGPL-3.0, or buy an Ultralytics Enterprise licence. A closed-source product needs the Enterprise licence (or a
  switch to an Apache / MIT detector stack).
- **MobileCLIP2** (`weights/mobileclip2_b.ts`, used only by the optional "Check with YOLOE" open-vocabulary path) comes
  from Apple's ml-mobileclip release, whose model licence allows research use. Leave it out of any commercial build.
- **OpenAI CLIP** weights (`weights/clip/ViT-B-32.pt`, `models/clip_int8/ViT-B-32_openai_*`): the code is MIT; OpenAI's
  model card lists deployed and surveillance uses as out of scope (guidance, not a licence term).
- **Sample videos** are NOT covered by any licence here: see `samples/SOURCES.md`.

## Software (Python packages)

| Package | Version | Licence | Used for |
|---|---|---|---|
| ultralytics | 8.4.174 | AGPL-3.0 | YOLO detection, pose, ByteTrack, YOLO-World, YOLOE |
| torch | 2.14.1+cpu | BSD-3-Clause (with Apache-2.0 parts) | neural network runtime |
| torchvision | 0.29.1+cpu | BSD-3-Clause | image ops for torch |
| opencv-python | 5.0.0.93 | Apache-2.0 | video reading / writing, drawing |
| numpy | 2.4.6 | BSD-3-Clause (and 0BSD, MIT, Zlib, CC0 parts) | numbers |
| scipy | 1.17.1 | BSD-3-Clause | numbers |
| pandas | 3.0.6 | BSD-3-Clause | tables in the dashboard |
| streamlit | 1.65.0 | Apache-2.0 | dashboard |
| altair | 6.3.0 | BSD-3-Clause | charts |
| matplotlib | 3.11.2 | Matplotlib licence (PSF-based) | evidence plots |
| open_clip_torch | 3.3.0 | MIT | CLIP search and the chat's question router |
| openvino | 2026.4.1 | Apache-2.0 | fast CPU inference |
| onnx | 1.23.2 | Apache-2.0 | model export (measured) |
| onnxruntime | 1.31.0 | MIT | model export (measured) |
| huggingface_hub | 2.2.0 | Apache-2.0 | open_clip model files (offline) |
| timm | 1.0.30 | Apache-2.0 | open_clip dependency |
| safetensors | 0.8.0 | Apache-2.0 | open_clip dependency |
| ftfy | 6.3.1 | Apache-2.0 | open_clip text cleaning |
| regex | 2026.9.29 | Apache-2.0 (and CNRI-Python) | open_clip tokeniser |
| pillow | 12.3.0 | MIT-CMU (HPND) | images |
| PyYAML | 6.0.3 | MIT | settings files |
| lap | 0.5.13 | BSD-2-Clause | ByteTrack matching |
| websockets | 17.2 | BSD-3-Clause | dashboard screenshot scripts only |

Outside Python: Windows speech (system voice for spoken alerts).

## Model weights

| File | Source | Licence / terms |
|---|---|---|
| `yolo11n.pt`, `yolo11s.pt`, `yolo11n-pose.pt` | Ultralytics releases (trained on COCO) | AGPL-3.0 |
| `models/yolov8s-worldv2.pt` | Ultralytics YOLO-World v2 | AGPL-3.0 |
| `models/yoloe-26s-seg.pt` | Ultralytics YOLOE | AGPL-3.0 |
| `models/ovdetect/*_openvino_model`, `models/quant/` | our exports of the files above | same as the source weights |
| `weights/mobileclip2_b.ts` | Apple ml-mobileclip (MobileCLIP2) | Apple model licence: research use |
| `weights/clip/ViT-B-32.pt`, `models/clip_int8/ViT-B-32_openai_*` | OpenAI CLIP via open_clip | MIT (code); see the model card note above |
| `models/clip_int8/ViT-B-32_laion2b_*` | OpenCLIP, trained on LAION-2B | MIT (weights released by OpenCLIP) |
| `training/runs/swimmer/weights/best.pt` (5-class posture model) | trained by the team with Ultralytics | AGPL-3.0 (Ultralytics) + CC BY 4.0 data credits below |
| `models/pool_person_s800` (person model, YOLO11s @800) | trained by the team on Colab with Ultralytics | AGPL-3.0 (Ultralytics) + CC BY 4.0 data credits below |

## The models download (GitHub release `models-v1`)

`python aquawatch/get_models.py` downloads `aquawatch-models-v1.zip` from the project's GitHub release: the
OpenVINO FP32 exports of `yolo11n.pt` and `yolo11s.pt` (Ultralytics, AGPL-3.0), of our 5-class posture model
`best.pt` (AGPL-3.0, trained on the CC BY 4.0 datasets credited below), and `yolo11n-pose.pt` (Ultralytics,
AGPL-3.0). With `--with-chat-model` it also fetches OpenAI CLIP ViT-B/32 from OpenAI's own address.

## Training datasets (CC BY 4.0: credit required)

All three are from Roboflow Universe under CC BY 4.0. Which model used which (from `datasets/swimmers/manifest.csv`
and `models/pool_person_s800/CREDITS.md`):

| Dataset | Author | Link | Used in |
|---|---|---|---|
| Swimming and Drowning Detection | University (university-g3h71) | https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection | posture model `best.pt` (12,365 images) and person model |
| drowning (swimming / drowning / treading_water) | children in buggy | https://universe.roboflow.com/children-in-buggy/drowning-i0nae | posture model `best.pt` (716 images) and person model |
| Drowning Detection And Prevention In Swimming Pools | yashwanee tetar | https://universe.roboflow.com/yashwanee-tetar-jm88u/drowning-detection-and-prevention-in-swimming-pools | person model only |

Downloaded but **not used**: "drowning detection" by Pool safety
(https://universe.roboflow.com/pool-safety/drowning-detection-otnme, CC BY 4.0): checked to be a pixel-identical
re-upload of the University set (40 of 40 sampled frames), so it was left out (`training/pool_datasets.yaml`).

CC BY 4.0 allows commercial use with credit, but the datasets' own images may show people who never agreed to a
commercial product; the rights of the original footage behind each Roboflow upload are not known.
