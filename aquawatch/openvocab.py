"""Open-vocabulary visual search over a site's footage with CLIP (AquaWatch).

    python openvocab.py build --db outputs/virtual_cctv/aquawatch.sqlite3
    python openvocab.py query --db outputs/virtual_cctv/aquawatch.sqlite3 "lifeguard jumping into the pool"

Queries like "child in a blue life jacket" or "big splash" do not match the detector's fixed classes, so this module
embeds the footage with a vision-language model (open_clip ViT-B-32, zero-shot, nothing trained by us) and ranks it
against the text:

  1. build : for every camera of the event index, cut a crop of every tracked box every `crop_every_s` (from
             stage1_tracking.jsonl, boxes scaled from processing pixels to the video) and take the whole frame every
             `frame_every_s`; embed both with the CLIP image tower. Saved as openvocab_index.npz + .json next to the
             index (cached: rebuilt only when a run or the settings change).
  2. search: the text (a few prompt templates, averaged) against every crop and frame. Three ideas on top of plain
             embedding retrieval (each can be switched off for the ablation in tools/eval_openvocab.py):
               * calibration (v2): the query similarity minus the similarity to neutral prompts ("a person in a
                 swimming pool"), standardised over the whole gallery (median / MAD z-score). A hit's confidence is
                 1 - n * P(Z >= z) (Bonferroni over the n items ranked). So "looks like any swimmer" does not count,
                 track and frame hits are on one scale, and a best hit that does not stand out from the gallery
                 gives "no confident match" (`min_confidence`) instead of always returning something.
                 (v1 used a per-image softmax against the neutral prompts; it saturated at 1.0 for some crop on
                 every query, even "a red car": see docs/RESULTS.md section M);
               * track aggregation: crops of one local track are pooled (mean of its best `track_topn` z-scores),
                 so a single noisy crop cannot win; the answer carries the track and the time span it matched;
               * fusion: track hits (people) and frame hits (scene: splash, black screen) are ranked together.
  3. answer: camera + video time + wall-clock time + track + an evidence JPG (the frame with the matched box) per hit.

The baseline is the standard pipeline: whole-frame CLIP embeddings ranked by raw cosine similarity (method
"baseline_frame"). Scores are model similarity, never verification: the answer says so.

Compact / fast variants (all off by default; measured for pre-registration rule R4, see `bench` and
tools/eval_openvocab.py):
  * `clip_int8`: dynamic INT8 quantization (torch) of the CLIP nn.Linear layers (the MLPs of both towers);
  * `embedding_dtype: int8`: the index stores int8 embeddings with one float32 scale per row (symmetric), half the
    size of float16; `convert` turns an existing index into int8 without re-embedding;
  * `pretrained: openai`: OpenAI CLIP ViT-B/32 from the local file weights/clip/ViT-B-32.pt (the file YOLO-World uses),
    never downloaded. The text encoder used by search() always follows the model recorded in the index.

    python openvocab.py build --db ... --out-dir outputs/virtual_cctv/ov_openai --pretrained openai --clip-int8
    python openvocab.py convert --src outputs/virtual_cctv --dst outputs/virtual_cctv/ov_laion_emb_int8
    python openvocab.py bench --db ... --pretrained openai --out bench.json

torch / open_clip / cv2 are imported inside functions only, so importing this module stays cheap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import cameras as camreg

# Defaults. An optional `openvocab:` section of config.yaml overrides them (load_settings), like reid.py.
OPENVOCAB_DEFAULTS: dict[str, Any] = {
    "model": "ViT-B-32",                 # open_clip architecture
    "pretrained": "openai",              # the local OpenAI file below (the CLIP YOLO-World needs anyway: one model on
                                         # disk); "laion2b_s34b_b79k" = LAION weights from the Hugging Face cache; or
                                         # the path of a checkpoint file. OpenAI won R4 (docs/RESULTS.md section N)
    "openai_weights": "weights/clip/ViT-B-32.pt",   # OpenAI CLIP ViT-B/32 (TorchScript, the file YOLO-World uses)
    "clip_int8": False,                  # dynamic INT8 quantization of CLIP's nn.Linear layers (torch, CPU)
    "embedding_dtype": "int8",           # how build() stores embeddings: "int8" (+ a scale per row; passed R4, index
                                         # 61 % smaller) or "float16"
    "threads": 2,                        # torch CPU threads (keeps the laptop responsive)
    "batch": 32,                         # images per forward pass
    "crop_every_s": 0.5,                 # one crop per track per this many seconds
    "frame_every_s": 0.5,                # one whole frame per camera per this many seconds
    "crop_pad": 0.15,                    # crop margin around the box, as a fraction of its size
    "min_box_h_px": 12.0,                # smaller boxes (processing pixels) are skipped
    "min_crop_px": 24,                   # crops are enlarged to at least this side (video pixels)
    "templates": ["a photo of {}.", "a cctv image of {}.", "{}"],
    "neutral_crop": ["a photo of a person in a swimming pool.", "a photo of a swimmer in the water.",
                     "a photo of pool water."],
    "neutral_frame": ["a photo of a crowded swimming pool.", "a cctv image of people swimming in a wave pool."],
    "track_topn": 3,                     # a track's score = mean of its best N crop scores
    "span_drop_z": 1.0,                  # matched span = crops within this many z of the track's best crop
    "frame_smooth": 1,                   # frame scores averaged over +-N neighbouring samples
    "min_confidence": 0.95,              # best hit's confidence below this: "no confident match" (= a 5 %
                                         # chance that pure noise passes: family-wise alpha 0.05)
    "top_k": 5,                          # hits returned
    # what Ask the Video uses. The frame baseline won R1 (19 held-out queries, OpenAI CLIP: hit@1 12, hit@5 18);
    # YOLOE / YOLO-World retrieve-then-detect ("ground_yoloe", "ground_yoloworld") reject every "not there" query but
    # find far fewer real ones (6 / 3 at hit@1) and take ~9 s per query on this CPU (docs/RESULTS.md sections M, N)
    "chat_method": "baseline_frame",
    "same_moment_s": 1.0,                # two frame hits of one camera closer than this are one hit
}
INDEX_KEYS = ("model", "pretrained", "crop_every_s", "frame_every_s", "crop_pad", "min_box_h_px", "min_crop_px")
EMBEDDING_DTYPES = ("float16", "int8")
INDEX_NAME = "openvocab_index"           # <dir>/openvocab_index.npz + .json
INDEX_VERSION = 1
# development texts for `bench` (in none of the evaluation query sets, see PREREGISTRATION_v3.md)
BENCH_TEXTS = ("a palm tree", "a person wearing sunglasses", "a green towel", "an airplane in the sky",
               "a crowd of people", "a yellow ball floating", "a woman with long hair", "a man in a white t-shirt",
               "an empty swimming pool", "a boat", "a building with windows", "a person waving", "a bird",
               "a fountain", "a clock", "a staircase", "two people talking", "a child with goggles",
               "a striped shirt", "a flag")
EVIDENCE_DIR = "openvocab_evidence"
METHODS = ("baseline_frame", "crops_raw", "crops_calibrated", "tracks", "fused")
CALIBRATED_METHODS = ("crops_calibrated", "tracks", "fused")    # their score is a confidence in [0, 1]
DISCLAIMER = ("Open-vocabulary match by CLIP similarity (zero-shot), not a verified identification: check the "
              "evidence image.")

_MODEL_CACHE: dict[tuple, Any] = {}       # (model, pretrained, clip_int8) -> (model, preprocess, tokenizer)
_TEXT_CACHE: dict[tuple, np.ndarray] = {}  # (model, pretrained, texts, clip_int8) -> unit vectors
_QUANT_FAILED: dict[tuple, str] = {}      # model keys whose INT8 quantization failed (float32 used instead)


def load_settings(overrides: Mapping[str, Any] | None = None, use_config: bool = True) -> dict:
    """OPENVOCAB_DEFAULTS, then config.yaml's optional `openvocab:` section, then the non-None overrides."""
    settings = dict(OPENVOCAB_DEFAULTS)
    if use_config:
        try:
            import utils
            settings.update(utils.load_config().get("openvocab") or {})
        except Exception:                                     # a broken config must not stop the search
            pass
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return settings


def index_paths(directory: str | Path) -> tuple[Path, Path]:
    """(<dir>/openvocab_index.npz, <dir>/openvocab_index.json)."""
    directory = Path(directory)
    return directory / f"{INDEX_NAME}.npz", directory / f"{INDEX_NAME}.json"


# ---------------------------------------------------------------------------- model (torch inside functions)

def model_key(settings: Mapping[str, Any]) -> tuple:
    """Cache key of a CLIP model: (architecture, weights, INT8 quantized?)."""
    return (settings["model"], settings["pretrained"], bool(settings.get("clip_int8")))


def encoder_settings(index: Mapping[str, Any] | None, settings: Mapping[str, Any]) -> dict:
    """`settings` with the model / weights / clip_int8 the index was embedded with (recorded in its json), so a query
    is encoded into the same embedding space as the gallery. Unchanged when the index records nothing."""
    out = dict(settings)
    recorded = ((index or {}).get("info") or {}).get("settings") or {}
    if recorded.get("model") and recorded.get("pretrained"):
        out.update(model=recorded["model"], pretrained=recorded["pretrained"],
                   clip_int8=bool(recorded.get("clip_int8", False)))
    return out


def weights_file(settings: Mapping[str, Any]) -> Path | None:
    """The local checkpoint file `pretrained` points to ("openai" = settings openai_weights), or None for an open_clip
    tag. "openai" never downloads: a missing file is an error."""
    pretrained = str(settings["pretrained"])
    if pretrained == "openai":
        path = camreg.resolve_path(settings.get("openai_weights") or OPENVOCAB_DEFAULTS["openai_weights"])
        if not path.is_file():
            raise FileNotFoundError(f"OpenAI CLIP weights not found at {path} (not downloading them)")
        return path
    path = camreg.resolve_path(pretrained)
    return path if path.is_file() else None


def _is_torchscript(path: Path) -> bool:
    """True for a TorchScript archive (the format of OpenAI's CLIP .pt files): a zip with a code/ folder."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as archive:
            return any("/code/" in name for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def _load_clip(settings: Mapping[str, Any]):
    """(float32 model, preprocess, tokenizer) on the CPU with `threads` torch threads. OpenAI TorchScript files go
    through open_clip.load_openai_model (it builds the QuickGELU model from the file); tags and other files through
    open_clip.create_model_and_transforms."""
    import open_clip
    import torch
    import cpu_limits
    cpu_limits.apply()
    torch.set_num_threads(int(settings.get("threads") or 2))
    path = weights_file(settings)
    if path is not None and _is_torchscript(path):
        model = open_clip.load_openai_model(str(path), precision="fp32", device="cpu")
        preprocess = open_clip.image_transform(model.visual.image_size, is_train=False,
                                               mean=model.visual.image_mean, std=model.visual.image_std)
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings["model"], pretrained=str(path) if path is not None else settings["pretrained"])
    model.eval()
    return model, preprocess, open_clip.get_tokenizer(settings["model"])


def quantize_linear(model):
    """Dynamic INT8 quantization of every nn.Linear (weights int8, activations quantized on the fly), in place.
    In open_clip's ViT these are the MLP layers of both towers; attention projections stay float32."""
    import warnings
    import torch
    with warnings.catch_warnings():                      # torch.ao.quantization prints deprecation warnings
        warnings.simplefilter("ignore")
        from torch.ao.quantization import quantize_dynamic
        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    # open_clip's encode_text asks the first MLP layer for its weight dtype (`.weight.dtype`); a quantized layer's
    # weight is a method, so it would crash. open_clip reads `int8_original_dtype` first (its own int8 hook).
    for module in model.modules():
        if "quantized" in type(module).__module__ and type(module).__name__ == "Linear":
            module.int8_original_dtype = torch.float32
    return model


def _model(settings: Mapping[str, Any]):
    """(model, preprocess, tokenizer), loaded once per process on the CPU with `threads` threads; INT8 quantized when
    `clip_int8`. If quantization fails, the float32 model is used and the failure is reported (never a crash)."""
    key = model_key(settings)
    if key not in _MODEL_CACHE:
        model, preprocess, tokenizer = _load_clip(settings)
        if key[2]:
            try:
                model = quantize_linear(model)
            except Exception as exc:                     # optional speed-up: report and continue in float32
                _QUANT_FAILED[key] = f"{type(exc).__name__}: {exc}"
                print(f"openvocab: clip_int8 failed ({_QUANT_FAILED[key]}); using float32")
        _MODEL_CACHE[key] = (model, preprocess, tokenizer)
    return _MODEL_CACHE[key]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def embed_images(images_bgr: list[np.ndarray], settings: Mapping[str, Any]) -> np.ndarray:
    """CLIP image embeddings (N x D, unit length) of BGR images."""
    import torch
    from PIL import Image
    model, preprocess, _ = _model(settings)
    out = []
    batch = int(settings.get("batch") or 32)
    with torch.no_grad():
        for start in range(0, len(images_bgr), batch):
            tensors = [preprocess(Image.fromarray(img[:, :, ::-1].copy())) for img in images_bgr[start:start + batch]]
            out.append(model.encode_image(torch.stack(tensors)).float().numpy())
    return _normalize(np.concatenate(out)) if out else np.zeros((0, 512), np.float32)


def embed_texts(texts: list[str], settings: Mapping[str, Any]) -> np.ndarray:
    """CLIP text embeddings (N x D, unit length); cached per process."""
    import torch
    key = (settings["model"], settings["pretrained"], tuple(texts), bool(settings.get("clip_int8")))
    if key not in _TEXT_CACHE:
        model, _, tokenizer = _model(settings)
        with torch.no_grad():
            _TEXT_CACHE[key] = _normalize(model.encode_text(tokenizer(texts)).float().numpy())
    return _TEXT_CACHE[key]


def text_vector(query: str, settings: Mapping[str, Any], templates: Iterable[str] | None = None) -> np.ndarray:
    """One query embedding: the normalised mean over the prompt templates."""
    templates = list(templates if templates is not None else settings["templates"])
    return _normalize(embed_texts([t.format(query) for t in templates], settings).mean(axis=0, keepdims=True))[0]


# ---------------------------------------------------------------------------- building the index

def _content_stamp(path: Path) -> str:
    """SHA-1 of a file's bytes ('' when missing): the cache stays valid when the file is rewritten unchanged."""
    import hashlib
    try:
        digest = hashlib.sha1()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return "sha1:" + digest.hexdigest()
    except OSError:
        return ""


def _file_stamp(path: Path) -> str:
    """Size + mtime of a file ('' when missing): cheap change detection for the cache."""
    try:
        stat = Path(path).stat()
        return f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return ""


def _read_stage1(run_dir: Path) -> tuple[dict, list[dict]]:
    """(meta, box rows) of stage1_tracking.jsonl."""
    meta, rows = {}, []
    path = run_dir / "stage1_tracking.jsonl"
    if not path.is_file():
        return meta, rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "meta" in row:
                meta = row["meta"]
            elif row.get("person_id") is not None:
                rows.append(row)
    return meta, rows


def select_crop_samples(rows: list[dict], every_s: float, min_box_h: float) -> list[dict]:
    """One box per track per `every_s` seconds (the first box of each time bin), big enough boxes only."""
    picked, seen = [], set()
    for row in sorted(rows, key=lambda r: (int(r["person_id"]), float(r["timestamp"]))):
        x1, y1, x2, y2 = (float(v) for v in row["bbox"])
        if y2 - y1 < min_box_h:
            continue
        key = (int(row["person_id"]), int(float(row["timestamp"]) // every_s))
        if key not in seen:
            seen.add(key)
            picked.append(row)
    return picked


def crop_box(frame: np.ndarray, box_video: list[float], pad: float, min_side: int) -> np.ndarray:
    """The padded box cut from the frame, enlarged to at least `min_side` pixels on each side."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box_video
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max(bw * (1 + 2 * pad), min_side) / 2
    half_h = max(bh * (1 + 2 * pad), min_side) / 2
    a, b = int(max(0, cx - half_w)), int(max(0, cy - half_h))
    c, d = int(min(w, cx + half_w)), int(min(h, cy + half_h))
    return frame[b:max(d, b + 1), a:max(c, a + 1)]


def embed_camera(camera: Mapping[str, Any], settings: Mapping[str, Any], echo=print) -> dict:
    """Crops + frames of one camera, embedded: {"crops": [meta], "crop_emb", "frames": [meta], "frame_emb", ...}."""
    import cv2
    cam_id = camera["camera_id"]
    run_dir = camreg.resolve_path(camera.get("out_dir") or "")
    video = camreg.resolve_path(camera.get("video") or "")
    meta, rows = _read_stage1(run_dir)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"{cam_id}: cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps") or 30.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc = meta.get("proc_size") or [width, height]
    sx, sy = width / float(proc[0]), height / float(proc[1])
    samples = select_crop_samples(rows, float(settings["crop_every_s"]), float(settings["min_box_h_px"]))
    by_frame: dict[int, list[dict]] = {}
    for row in samples:
        by_frame.setdefault(int(row["frame"]), []).append(row)
    frame_step = max(1, int(round(float(settings["frame_every_s"]) * fps)))
    crops, crop_imgs, frames, frame_imgs = [], [], [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        if index % frame_step == 0:
            frames.append({"camera_id": cam_id, "frame": index, "t": round(t, 3)})
            frame_imgs.append(frame)
        for row in by_frame.get(index, []):
            x1, y1, x2, y2 = (float(v) for v in row["bbox"])
            box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            crops.append({"camera_id": cam_id, "track_id": int(row["person_id"]), "frame": index,
                          "t": round(float(row["timestamp"]), 3), "box": [round(v, 1) for v in box]})
            crop_imgs.append(crop_box(frame, box, float(settings["crop_pad"]), int(settings["min_crop_px"])))
        index += 1
    cap.release()
    echo(f"  {cam_id}: {len(crops)} track crops, {len(frames)} frames ({index} video frames)")
    return {"crops": crops, "crop_emb": embed_images(crop_imgs, settings),
            "frames": frames, "frame_emb": embed_images(frame_imgs, settings)}


def build(db_path: str | Path, settings: Mapping[str, Any] | None = None, out_dir: str | Path | None = None,
          force: bool = False, echo=print) -> dict:
    """Embed every camera of the event index; cameras whose run and video are unchanged are reused from the cache."""
    from event_index import EventIndex
    started = time.time()
    settings = load_settings(settings)
    dtype = str(settings.get("embedding_dtype") or "float16")
    if dtype not in EMBEDDING_DTYPES:
        raise ValueError(f"embedding_dtype must be one of {EMBEDDING_DTYPES}, not {dtype!r}")
    out_dir = Path(out_dir) if out_dir else Path(db_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = index_paths(out_dir)
    old = load_index(out_dir) if not force else None
    cams = EventIndex(db_path).list_cameras()
    keyed = {k: settings[k] for k in INDEX_KEYS}
    if settings.get("clip_int8"):                         # recorded only when on: older float32 indexes stay reusable
        keyed["clip_int8"] = True
    parts = []
    for cam in cams:
        run_dir = camreg.resolve_path(cam.get("out_dir") or "")
        # stage1 by CONTENT: run.py rewrites the file (new mtime, same bytes) on every --reuse run, which made every
        # ingest re-embed all cameras with CLIP (bug hunt 2026-10-11, multicam low)
        stamp = {"stage1": _content_stamp(run_dir / "stage1_tracking.jsonl"),
                 "video": _file_stamp(camreg.resolve_path(cam.get("video") or ""))}
        reuse = (old and old["info"].get("settings") == keyed and old["info"].get("version") == INDEX_VERSION
                 and (old["info"].get("stamps") or {}).get(cam["camera_id"]) == stamp
                 # int8 rows are never re-used for a float16 index (their rounding error would be kept silently)
                 and (old["info"].get("embedding_dtype") or "float16") in ("float16", dtype))
        if reuse:
            echo(f"  {cam['camera_id']}: unchanged, reused")
            part = _slice(old, cam["camera_id"])
        else:
            part = embed_camera(cam, settings, echo=echo)
            failed = _QUANT_FAILED.get(model_key(settings)) if settings.get("clip_int8") else None
            if failed:                                    # never label float32 embeddings as int8
                raise RuntimeError(f"clip_int8 requested but quantization failed: {failed}")
        part["stamp"] = stamp
        parts.append((cam["camera_id"], part))
    crops = [c for _, p in parts for c in p["crops"]]
    frames = [f for _, p in parts for f in p["frames"]]
    dim = next((p["crop_emb"].shape[1] for _, p in parts if len(p["crop_emb"])), 512)
    crop_emb = np.concatenate([p["crop_emb"] for _, p in parts if len(p["crop_emb"])] or [np.zeros((0, dim))])
    frame_emb = np.concatenate([p["frame_emb"] for _, p in parts if len(p["frame_emb"])] or [np.zeros((0, dim))])
    info = {"version": INDEX_VERSION, "settings": keyed, "db": Path(db_path).name, "built_at":
            datetime.now().isoformat(timespec="seconds"), "stamps": {cid: p["stamp"] for cid, p in parts},
            "crops": crops, "frames": frames, "build_s": round(time.time() - started, 1)}
    save_index(out_dir, crop_emb, frame_emb, info, dtype)
    echo(f"open-vocabulary index: {len(crops)} crops + {len(frames)} frames ({dtype}) -> {npz_path}")
    return {"crops": len(crops), "frames": len(frames), "path": str(npz_path), "embedding_dtype": dtype,
            "bytes": index_bytes(out_dir)}


def quantize_rows(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per row: (int8 values, float32 scale per row) with row ~= values * scale."""
    matrix = np.asarray(matrix, dtype=np.float32)
    scale = (np.abs(matrix).max(axis=1) / 127.0 if len(matrix) else np.zeros(0)).astype(np.float32)
    safe = np.where(scale > 0, scale, 1.0)[:, None]                   # zero rows stay zero
    values = np.clip(np.rint(matrix / safe), -127, 127).astype(np.int8)
    return values, scale


def dequantize_rows(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """float32 rows back from int8 values and their per-row scales, re-scaled to unit length (scores stay cosines)."""
    return _normalize(values.astype(np.float32) * np.asarray(scale, np.float32)[:, None])


def save_index(directory: str | Path, crop_emb: np.ndarray, frame_emb: np.ndarray, info: dict,
               dtype: str = "float16") -> Path:
    """Writes <dir>/openvocab_index.npz (+ .json with info); `dtype` "float16" or "int8" (+ crop_scale / frame_scale)
    is recorded in the json as embedding_dtype."""
    if dtype not in EMBEDDING_DTYPES:
        raise ValueError(f"embedding_dtype must be one of {EMBEDDING_DTYPES}, not {dtype!r}")
    npz_path, json_path = index_paths(directory)
    Path(directory).mkdir(parents=True, exist_ok=True)
    if dtype == "int8":
        (cq, cs), (fq, fs) = quantize_rows(crop_emb), quantize_rows(frame_emb)
        np.savez_compressed(npz_path, crop_emb=cq, crop_scale=cs, frame_emb=fq, frame_scale=fs)
    else:
        np.savez_compressed(npz_path, crop_emb=np.asarray(crop_emb).astype(np.float16),
                            frame_emb=np.asarray(frame_emb).astype(np.float16))
    json_path.write_text(json.dumps(dict(info, embedding_dtype=dtype)), encoding="utf-8")
    return npz_path


def index_bytes(directory: str | Path) -> dict:
    """Sizes on disk of an index: {"npz": bytes, "json": bytes}."""
    return {kind: (path.stat().st_size if path.is_file() else 0)
            for kind, path in zip(("npz", "json"), index_paths(directory))}


def convert_index(src_dir: str | Path, dst_dir: str | Path, dtype: str = "int8", echo=print) -> dict:
    """Copies the index of `src_dir` to `dst_dir` with its embeddings stored as `dtype`, without re-embedding."""
    index = load_index(src_dir)
    if index is None:
        raise FileNotFoundError(f"no open-vocabulary index in {src_dir}")
    info = dict(index["info"], converted_from=str(Path(src_dir)),
                converted_at=datetime.now().isoformat(timespec="seconds"))
    info.pop("embedding_dtype", None)
    save_index(dst_dir, index["crop_emb"], index["frame_emb"], info, dtype)
    sizes = index_bytes(dst_dir)
    echo(f"converted {len(index['crops'])} crops + {len(index['frames'])} frames to {dtype} -> "
         f"{index_paths(dst_dir)[0]} ({sizes['npz']} bytes)")
    return {"path": str(index_paths(dst_dir)[0]), "embedding_dtype": dtype, "bytes": sizes}


def _slice(index: Mapping[str, Any], camera_id: str) -> dict:
    """The rows of one camera from a loaded index (for cache reuse)."""
    ci = [i for i, c in enumerate(index["crops"]) if c["camera_id"] == camera_id]
    fi = [i for i, f in enumerate(index["frames"]) if f["camera_id"] == camera_id]
    return {"crops": [index["crops"][i] for i in ci], "crop_emb": index["crop_emb"][ci],
            "frames": [index["frames"][i] for i in fi], "frame_emb": index["frame_emb"][fi]}


def load_index(directory: str | Path) -> dict | None:
    """{"crops", "crop_emb", "frames", "frame_emb", "info"} (float32 embeddings, from a float16 or an int8 index) or
    None when there is no (readable) index. An index without embedding_dtype in its json is float16."""
    npz_path, json_path = index_paths(directory)
    if not (npz_path.is_file() and json_path.is_file()):
        return None
    try:
        info = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path) as arrays:                # closed at once (Windows keeps open files locked)
            stored = "int8" if "crop_scale" in arrays.files else "float16"
            emb = {}
            for name in ("crop", "frame"):
                values = arrays[f"{name}_emb"]
                emb[name] = (dequantize_rows(values, arrays[f"{name}_scale"]) if stored == "int8"
                             else values.astype(np.float32))
        info.setdefault("embedding_dtype", stored)
        return {"crops": info["crops"], "frames": info["frames"], "info": info,
                "crop_emb": emb["crop"], "frame_emb": emb["frame"]}
    except (OSError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------- scoring (NumPy only)

def neutral_margin(sim_query: np.ndarray, sim_neutral: np.ndarray) -> np.ndarray:
    """How much more an image looks like the query than like the neutral prompts (cosine difference)."""
    return sim_query - sim_neutral.mean(axis=1)


def robust_z(values: np.ndarray, min_items: int = 5) -> np.ndarray:
    """(x - median) / (1.4826 * MAD) over the gallery; all zeros when there are too few items to judge."""
    values = np.asarray(values, dtype=float)
    if len(values) < min_items:
        return np.zeros_like(values)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return (values - median) / max(mad, 1e-6)


def log_p_adjusted(z: float, n: int) -> float:
    """log(n * P(Z >= z)) for a standard normal Z, capped at 0: how likely a gallery of n items holds an item this far
    above the rest by chance alone (Bonferroni). Very negative = stands out clearly."""
    tail = 0.5 * math.erfc(float(z) / math.sqrt(2.0))
    return min(0.0, math.log(max(n, 1)) + math.log(max(tail, 1e-300)))


def _rank(hits: list[dict]) -> list[dict]:
    """Most surprising first (log_p), then the highest z."""
    return sorted(hits, key=lambda h: (h.get("log_p", 0.0), -h.get("z", h["score"])))


def _confident(hits: list[dict], n: int) -> list[dict]:
    """Adds log_p and score = confidence = 1 - n * P(Z >= z) (0 = could be chance, 1 = clearly stands out)."""
    for h in hits:
        h["log_p"] = log_p_adjusted(h["z"], n)
        h["score"] = 1.0 - math.exp(h["log_p"])
    return hits


def _in_scope(item: Mapping[str, Any], cams: Iterable[str] | None, window: tuple | None) -> bool:
    """Camera and video-time filter: (start, end) seconds (either may be None), or {camera_id: (start, end)}."""
    if cams and item["camera_id"] not in cams:
        return False
    if isinstance(window, Mapping):
        window = window.get(item["camera_id"])
    if window:
        start, end = window
        if (start is not None and item["t"] < start) or (end is not None and item["t"] > end):
            return False
    return True


def aggregate_tracks(crops: list[dict], z: np.ndarray, topn: int, span_drop_z: float) -> list[dict]:
    """One hit per (camera, track): its statistic = mean of its best `topn` crop z-scores, then standardised across
    tracks (z); time = its best crop; span = its crops within `span_drop_z` of the best."""
    groups: dict[tuple, list[int]] = {}
    for i, c in enumerate(crops):
        groups.setdefault((c["camera_id"], c["track_id"]), []).append(i)
    hits = []
    for (cam, track), idx in groups.items():
        s = z[idx]
        order = np.argsort(-s)
        best = idx[int(order[0])]
        span = [crops[i]["t"] for i in idx if z[i] >= z[best] - span_drop_z]
        hits.append({"kind": "track", "camera_id": cam, "track_id": track, "pooled": float(s[order[:topn]].mean()),
                     "t": crops[best]["t"], "frame": crops[best]["frame"], "box": crops[best]["box"],
                     "span": [min(span), max(span)], "n_crops": len(idx)})
    if hits:
        for h, value in zip(hits, robust_z(np.array([h["pooled"] for h in hits]))):
            h["z"] = float(value)
    return _rank(_confident(hits, len(hits)))


def frame_hits(frames: list[dict], scores: np.ndarray, smooth: int, same_moment_s: float,
               normalize: bool = True) -> list[dict]:
    """Frame hits: scores smoothed over +-`smooth` samples per camera, standardised over all frames (normalize), one
    hit per camera and moment (the best frame within `same_moment_s`)."""
    by_cam: dict[str, list[int]] = {}
    for i, f in enumerate(frames):
        by_cam.setdefault(f["camera_id"], []).append(i)
    smoothed = np.array(scores, dtype=float)
    for idx in by_cam.values():
        idx = sorted(idx, key=lambda i: frames[i]["t"])
        if smooth > 0 and len(idx) > 1:
            kernel = np.ones(2 * smooth + 1) / (2 * smooth + 1)
            smoothed[idx] = np.convolve(np.pad(smoothed[idx], smooth, mode="edge"), kernel, mode="valid")
    z = robust_z(smoothed) if normalize else smoothed
    hits = []
    for j in np.argsort(-z):
        f = frames[int(j)]
        if any(h["camera_id"] == f["camera_id"] and abs(h["t"] - f["t"]) < same_moment_s for h in hits):
            continue
        hits.append({"kind": "frame", "camera_id": f["camera_id"], "track_id": None, "z": float(z[int(j)]),
                     "score": float(z[int(j)]), "t": f["t"], "frame": f["frame"], "box": None, "span": [f["t"], f["t"]]})
    return _rank(_confident(hits, len(frames))) if normalize else hits


def search(index: Mapping[str, Any], query: str, settings: Mapping[str, Any] | None = None, method: str = "fused",
           cams: Iterable[str] | None = None, window: tuple | Mapping | None = None, top_k: int | None = None,
           text_fn=None) -> list[dict]:
    """Ranked hits for a text query (see METHODS). `text_fn(list[str]) -> unit vectors` replaces CLIP in tests.

    Calibrated methods give every hit `score` = confidence in [0, 1] (1 - Bonferroni p of its z-score over the
    gallery), so track and frame hits are comparable and a weak best hit can be rejected (min_confidence)."""
    settings = encoder_settings(index, load_settings(settings, use_config=settings is None))
    top_k = int(top_k or settings["top_k"])
    cams = list(cams) if cams else None
    encode = text_fn or (lambda texts: embed_texts(texts, settings))

    def qvec(templates):
        return _normalize(encode([t.format(query) for t in templates]).mean(axis=0, keepdims=True))[0]

    def scoped(items, emb):
        keep = [i for i, it in enumerate(items) if _in_scope(it, cams, window)]
        return [items[i] for i in keep], emb[keep] if len(keep) else emb[:0]

    if method == "baseline_frame":                        # plain embedding retrieval: frames, raw cosine, one template
        frames, emb = scoped(index["frames"], index["frame_emb"])
        if not frames:
            return []
        return frame_hits(frames, emb @ qvec(["{}"]), 0, float(settings["same_moment_s"]), normalize=False)[:top_k]
    q = qvec(settings["templates"])
    hits: list[dict] = []
    crops, cemb = scoped(index["crops"], index["crop_emb"])
    if crops and method == "crops_raw":                   # ablation: crops, raw cosine, no calibration / pooling
        hits = sorted(({"kind": "crop", "camera_id": c["camera_id"], "track_id": c["track_id"], "score": float(s),
                        "t": c["t"], "frame": c["frame"], "box": c["box"], "span": [c["t"], c["t"]]}
                       for c, s in zip(crops, cemb @ q)), key=lambda h: -h["score"])
    elif crops and method in ("crops_calibrated", "tracks", "fused"):
        z = robust_z(neutral_margin(cemb @ q, cemb @ encode(list(settings["neutral_crop"])).T))
        if method == "crops_calibrated":                  # ablation: calibrated crops, no track pooling
            hits = _rank(_confident([{"kind": "crop", "camera_id": c["camera_id"], "track_id": c["track_id"],
                                      "z": float(v), "t": c["t"], "frame": c["frame"], "box": c["box"],
                                      "span": [c["t"], c["t"]]} for c, v in zip(crops, z)], len(crops)))
        else:
            hits = aggregate_tracks(crops, z, int(settings["track_topn"]), float(settings["span_drop_z"]))
    if method == "fused":
        frames, femb = scoped(index["frames"], index["frame_emb"])
        if frames:
            margin = neutral_margin(femb @ q, femb @ encode(list(settings["neutral_frame"])).T)
            hits = _rank(hits + frame_hits(frames, margin, int(settings["frame_smooth"]),
                                           float(settings["same_moment_s"])))
    return hits[:top_k]


# ---------------------------------------------------------------------------- question text and evidence

_LEAD = re.compile(r"^(?:(?:please|can you|could you|ok(?:ay)?|hey)[, ]+)*(?:find|search(?: for)?|look for|locate|"
                   r"show(?: me)?|where (?:is|was|are|were)|when (?:is|was|did)|is there|are there|was there|"
                   r"were there|did you see|do you see|have you seen|any|which cameras? (?:saw|shows?|showed|has|had))"
                   r"\s+", re.I)
_TAIL = re.compile(r"\s+(?:in|on|across|from) (?:the )?(?:footage|video|videos|cameras?|clip|pool area)\b.*$|"
                   r"\s+(?:ok|okay|alright|fine|safe)$", re.I)


def query_text(question: str, drop: Iterable[str] = ()) -> str:
    """The visual phrase of a question: 'Is there a child in a blue life jacket on cam02?' -> 'a child in a blue
    life jacket'. `drop` = substrings to remove first (camera tokens, time / place phrases)."""
    text = re.sub(r"\s+", " ", str(question or "")).strip().rstrip("?!. ")
    for part in drop:
        if part:
            text = re.sub(re.escape(part), " ", text, flags=re.I)
    text = re.sub(r"\s+(?:on|at|in|from|near|by)(?:\s+the)?$", "", re.sub(r"\s+", " ", text).strip()).strip()
    text = _LEAD.sub("", text).strip()
    text = re.sub(r"^(?:is|was|are|were|did|does|do)\s+", "", text, flags=re.I).strip()   # "is the swimmer ... ok"
    text = _TAIL.sub("", text).strip().rstrip(",.;:")
    return text or str(question or "").strip()


def write_evidence(hit: Mapping[str, Any], camera: Mapping[str, Any], out_dir: str | Path, query: str) -> str | None:
    """The hit's frame with its box (if any) and the query as caption -> JPG; returns the path relative to the
    project root, or None when the frame cannot be read."""
    import cv2
    video = camreg.resolve_path(camera.get("video") or "")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(hit["frame"]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if hit.get("box"):
        x1, y1, x2, y2 = (int(v) for v in hit["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    label = f"{hit['camera_id']} {hit['t']:.1f}s match {hit['score']:.2f}: {query}"[:90]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(frame, label.encode("ascii", "replace").decode("ascii"), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    folder = Path(out_dir) / EVIDENCE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(f"{query}|{hit['camera_id']}|{hit['track_id']}|{hit['frame']}".encode()).hexdigest()[:12]
    path = folder / f"{hit['camera_id']}_{hit['frame']}_{tag}.jpg"
    cv2.imwrite(str(path), frame)
    try:
        return path.resolve().relative_to(camreg.ROOT).as_posix()
    except ValueError:
        return str(path)


def wall_text(camera: Mapping[str, Any], t: float) -> str | None:
    """Wall-clock ISO text of video time t on a camera (start_time + t, like cameras.wall_clock and the event index;
    offset_s only places a camera on the site timeline), or None."""
    try:
        start = datetime.fromisoformat(str(camera.get("start_time")))
    except (TypeError, ValueError):
        return None
    return camreg.format_wall(start + timedelta(seconds=float(t)))


# ---------------------------------------------------------------------------- benchmark (float32 vs INT8 CLIP)

def _bench_frames(db_path: str | Path, count: int) -> list[np.ndarray]:
    """`count` whole frames spread evenly over the cameras of the event index (BGR)."""
    import cv2
    from event_index import EventIndex
    cams = EventIndex(db_path).list_cameras()
    per_cam = max(1, count // max(1, len(cams)))
    frames = []
    for cam in cams:
        cap = cv2.VideoCapture(str(camreg.resolve_path(cam.get("video") or "")))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for k in range(per_cam):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * total / per_cam))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
    return frames[:count]


def _tensor_bytes(state: Mapping[str, Any]) -> int:
    """Bytes of every tensor in a state_dict (packed INT8 weights included: they are stored as (weight, bias))."""
    import torch
    total = 0
    for value in state.values():
        for item in (value if isinstance(value, (tuple, list)) else (value,)):
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
    return total


def _measure_clip(model, preprocess, tokenizer, images: list[np.ndarray], texts: Iterable[str], batch: int) -> dict:
    """Size, image throughput and text latency of one loaded CLIP model, plus its embeddings (for agreement)."""
    import io
    import torch
    from PIL import Image
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    started = time.perf_counter()
    tensors = torch.stack([preprocess(Image.fromarray(img[:, :, ::-1].copy())) for img in images])
    prep_s = time.perf_counter() - started
    with torch.no_grad():
        model.encode_image(tensors[:8])                                  # warm-up (not timed)
        model.encode_text(tokenizer(["warm up"]))
        started = time.perf_counter()
        image_emb = torch.cat([model.encode_image(tensors[i:i + batch]) for i in range(0, len(tensors), batch)])
        image_s = time.perf_counter() - started
        text_ms, text_emb = [], []
        for text in texts:                                               # one query at a time, as in a search
            started = time.perf_counter()
            text_emb.append(model.encode_text(tokenizer([text])))
            text_ms.append((time.perf_counter() - started) * 1000)
    return {"state_dict_bytes_serialized": len(buffer.getvalue()), "state_dict_tensor_bytes": _tensor_bytes(
                model.state_dict()),
            "images": len(images), "preprocess_images_per_s": round(len(images) / prep_s, 2),
            "image_forward_images_per_s": round(len(images) / image_s, 2),
            "image_end_to_end_images_per_s": round(len(images) / (prep_s + image_s), 2),
            "text_ms_median": round(float(np.median(text_ms)), 2), "text_ms_max": round(float(np.max(text_ms)), 2),
            "texts": len(text_ms),
            "_image_emb": _normalize(image_emb.float().numpy()), "_text_emb": _normalize(torch.cat(text_emb).float().numpy())}


def bench(db_path: str | Path, settings: Mapping[str, Any] | None = None, n_images: int = 96,
          save_dir: str | Path | None = None, echo=print) -> dict:
    """float32 vs dynamic-INT8 CLIP on this CPU: model size, image throughput (real frames of the site), text latency
    per query (BENCH_TEXTS, none of them an evaluation query) and how close the INT8 embeddings stay to float32."""
    import copy
    import platform
    import torch
    settings = dict(load_settings(settings), clip_int8=False)
    batch = int(settings.get("batch") or 32)
    started = time.perf_counter()
    model, preprocess, tokenizer = _load_clip(settings)
    load_s = time.perf_counter() - started
    images = _bench_frames(db_path, n_images)
    echo(f"bench: {settings['model']} / {settings['pretrained']}, {len(images)} frames, {torch.get_num_threads()} "
         f"threads, quantized engine {torch.backends.quantized.engine}")
    fp32 = _measure_clip(model, preprocess, tokenizer, images, BENCH_TEXTS, batch)
    started = time.perf_counter()
    qmodel = quantize_linear(copy.deepcopy(model))
    quant_s = time.perf_counter() - started
    linear = sum(isinstance(m, torch.nn.Linear) for m in model.modules())
    quantized = sum("quantized" in type(m).__module__ and type(m).__name__ == "Linear" for m in qmodel.modules())
    int8 = _measure_clip(qmodel, preprocess, tokenizer, images, BENCH_TEXTS, batch)
    image_cos = np.sum(fp32["_image_emb"] * int8["_image_emb"], axis=1)
    text_cos = np.sum(fp32["_text_emb"] * int8["_text_emb"], axis=1)
    best_fp = np.argmax(fp32["_text_emb"] @ fp32["_image_emb"].T, axis=1)
    best_q = np.argmax(int8["_text_emb"] @ int8["_image_emb"].T, axis=1)
    result = {
        "model": settings["model"], "pretrained": settings["pretrained"], "torch": torch.__version__,
        "threads": torch.get_num_threads(), "quantized_engine": torch.backends.quantized.engine,
        "cpu": platform.processor(), "batch": batch, "load_s": round(load_s, 2), "quantize_s": round(quant_s, 2),
        "linear_layers": int(linear), "linear_layers_quantized": int(quantized),
        "fp32": {k: v for k, v in fp32.items() if not k.startswith("_")},
        "int8": {k: v for k, v in int8.items() if not k.startswith("_")},
        "agreement": {"image_cosine_mean": round(float(image_cos.mean()), 5),
                      "image_cosine_min": round(float(image_cos.min()), 5),
                      "text_cosine_mean": round(float(text_cos.mean()), 5),
                      "text_cosine_min": round(float(text_cos.min()), 5),
                      "same_best_frame_texts": int(np.sum(best_fp == best_q)), "texts": len(BENCH_TEXTS)}}
    if save_dir:                                          # the INT8 weights on disk (not needed at run time)
        folder = Path(save_dir)
        folder.mkdir(parents=True, exist_ok=True)
        tag = Path(str(settings["pretrained"])).stem
        path = folder / f"{settings['model']}_{tag}_int8_state_dict.pt"
        torch.save(qmodel.state_dict(), path)
        result["int8_file"] = {"path": str(path), "bytes": path.stat().st_size}
        source = weights_file(settings)
        if source is not None:
            result["fp32_source_file"] = {"path": str(source), "bytes": source.stat().st_size}
    return result


# ---------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    """build / convert / query / bench from the command line."""
    parser = argparse.ArgumentParser(description="AquaWatch open-vocabulary search (CLIP)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="embed every camera of an event index")
    b.add_argument("--db", required=True)
    b.add_argument("--force", action="store_true", help="re-embed even unchanged cameras")
    b.add_argument("--out-dir", help="index folder (default: the folder of --db)")
    b.add_argument("--pretrained", help="open_clip weights tag, 'openai' (local OpenAI file) or a checkpoint path")
    b.add_argument("--clip-int8", action="store_true", help="dynamic INT8 quantization of CLIP's linear layers")
    b.add_argument("--emb-int8", action="store_true", help="store int8 embeddings (+ a scale per row)")
    c = sub.add_parser("convert", help="store an existing index's embeddings in another dtype (no re-embedding)")
    c.add_argument("--src", required=True, help="folder of the existing index")
    c.add_argument("--dst", required=True, help="folder of the converted copy")
    c.add_argument("--dtype", default="int8", choices=EMBEDDING_DTYPES)
    q = sub.add_parser("query", help="search the index")
    q.add_argument("--db", required=True)
    q.add_argument("--index-dir", help="index folder (default: the folder of --db)")
    q.add_argument("--method", default="fused", choices=METHODS)
    q.add_argument("text", nargs="+")
    m = sub.add_parser("bench", help="float32 vs INT8 CLIP: size, image throughput, text latency, agreement")
    m.add_argument("--db", required=True, help="event index (its videos give the test frames)")
    m.add_argument("--pretrained", help="open_clip weights tag, 'openai' or a checkpoint path")
    m.add_argument("--images", type=int, default=96)
    m.add_argument("--save-dir", help="also save the INT8 state_dict here (e.g. models/clip_int8)")
    m.add_argument("--out", help="write the result as JSON")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        started = time.time()
        build(args.db, {"pretrained": args.pretrained, "clip_int8": True if args.clip_int8 else None,
                        "embedding_dtype": "int8" if args.emb_int8 else None}, out_dir=args.out_dir, force=args.force)
        print(f"done in {time.time() - started:.1f} s")
        return 0
    if args.cmd == "convert":
        started = time.time()
        convert_index(args.src, args.dst, args.dtype)
        print(f"done in {time.time() - started:.2f} s")
        return 0
    if args.cmd == "bench":
        result = bench(args.db, {"pretrained": args.pretrained}, n_images=args.images, save_dir=args.save_dir)
        print(json.dumps(result, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
            print(f"wrote {args.out}")
        return 0
    index = load_index(args.index_dir or Path(args.db).parent)
    if index is None:
        print("no open-vocabulary index: run 'python openvocab.py build --db ...' first")
        return 1
    for text in args.text:
        started = time.time()
        hits = search(index, text, method=args.method)
        print(f"Q: {text}  ({(time.time() - started) * 1000:.0f} ms)")
        for h in hits:
            who = f"track {h['track_id']}" if h["track_id"] is not None else "whole frame"
            print(f"  {h['camera_id']} t={h['t']:.1f}s {who} score {h['score']:.3f} span {h['span']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
