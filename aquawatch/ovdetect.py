"""Open-vocabulary DETECTION for Ask the Video: YOLO-World v2 and YOLOE-26s (AquaWatch).

    python ovdetect.py ground --db outputs/virtual_cctv/aquawatch.sqlite3 --backend yoloe "a person swimming"
    python ovdetect.py vocab  --db outputs/virtual_cctv/aquawatch.sqlite3 --backend yoloe
    python ovdetect.py export --backend yoloe --int8

openvocab.py ranks whole frames by CLIP similarity: it finds the right moment but cannot point at the thing and cannot
say "not there". This module adds a detector that takes the query text as its class name:

  1. ground (retrieve, then detect): stage 1 = the CLIP frame baseline (openvocab.search, method "baseline_frame")
     picks K = 12 candidate frames; stage 2 = the open-vocabulary detector runs on those frames with the query text as
     its ONLY class (640 px, confidence 0.25). One hit per candidate frame = its best box; frames without a box
     >= 0.25 are dropped, so "nothing anywhere" gives [] = "no confident match".
  2. vocab: a fixed vocabulary (26 terms, PREREGISTRATION_v3.md) detected on every camera every 0.5 s and saved as
     <db dir>/ovdetect_vocab_<backend>.json; vocab_search() answers a query that is exactly one term from it.
  3. export: the detector with the vocabulary baked in, exported to OpenVINO (optionally INT8, calibrated on frames of
     OTHER videos than the demo clip) for faster CPU inference, capped at 2 threads on the CPU. Only useful for the
     vocabulary index: grounding needs the free query text as its class, which an exported model cannot change.
     `python ovdetect.py agree ...` is the PREREGISTRATION_v3 R5 check (exported model vs PyTorch FP32, F1 >= 0.98).
     First results (2026-10-10, outputs/virtual_cctv/ovdetect_agree_*.json): FP32 exports F1 1.0 for both backends,
     INT8 exports with Ultralytics' default NNCF recipe failed R5 for both (YOLOE F1 0.599, YOLO-World F1 0.850).

Colour order (checked, see `python ovdetect.py check-colour`): both backends take the BGR numpy frame from cv2 and the
Ultralytics predictor flips it to RGB before the network, as in training. The "red/blue swapped" look of YOLOE's
result.plot() is its segmentation-mask overlay (class colour, 50 % alpha) painted over the person, not a channel swap.

All defaults below were fixed in data/virtual_cctv/PREREGISTRATION_v3.md before any result existed. An optional
`ovdetect:` section of config.yaml overrides them (load_settings). torch / ultralytics / cv2 / openvino are imported
inside functions only, so importing this module stays cheap.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

import cameras as camreg

ROOT = camreg.ROOT

# Defaults. An optional `ovdetect:` section of config.yaml overrides them (load_settings), like openvocab.py / reid.py.
OVDETECT_DEFAULTS: dict[str, Any] = {
    "backends": {"yoloworld": "models/yolov8s-worldv2.pt", "yoloe": "models/yoloe-26s-seg.pt"},
    # text encoders, loaded from these files only (never downloaded): YOLO-World = OpenAI CLIP ViT-B/32,
    # YOLOE-26 = MobileCLIP2-B TorchScript
    "text_weights": {"yoloworld": "weights/clip/ViT-B-32.pt", "yoloe": "weights/mobileclip2_b.ts"},
    "yolo_config_dir": "../.ultralytics",  # Ultralytics settings folder (relative to the project root)
    "threads": 2,                          # torch / OpenVINO CPU threads (keeps the laptop responsive)
    "below_normal_priority": True,         # the CLI lowers its own process priority (the venv's python.exe is a
                                           # launcher: a priority set on it from outside may not reach the interpreter)
    "openvino_device": "CPU",              # exported models: Ultralytics' "AUTO" is replaced by this device
    "imgsz": 640,                          # detector input size (pre-registered)
    "conf": 0.25,                          # box confidence threshold (pre-registered, the Ultralytics default)
    "nms_iou": 0.7,                        # NMS IoU (the Ultralytics default). With Ultralytics' default nms=None
                                           # YOLOE-26 also runs its one-to-many head + NMS (checked: end2end False)
    "max_det": 300,                        # boxes per frame (the Ultralytics default)
    # class-agnostic NMS per backend = Ultralytics' own predict defaults (YOLOE.predict sets agnostic_nms=True,
    # YOLOWorld does not). Passed explicitly, so an exported model (loaded as a plain YOLO) post-processes the same way
    "agnostic_nms": {"yoloworld": False, "yoloe": True},
    "candidates_k": 12,                    # K candidate frames from the CLIP frame baseline (pre-registered)
    "top_k": 5,                            # hits returned by ground() / vocab_search()
    "same_moment_s": 1.0,                  # vocab_search: one hit per camera per this many seconds
    "every_s": 0.5,                        # vocabulary index: one frame per camera per this many seconds
    "vocabulary": ["person", "child", "lifeguard", "swimmer", "life jacket", "inflatable ring", "rescue tube",
                   "swim cap", "goggles", "umbrella", "ladder", "rope", "buoy", "splash", "towel", "chair", "bag",
                   "backpack", "hat", "sunglasses", "ball", "phone", "bottle", "dog", "car", "bicycle"],
    "irregular_plurals": {"children": "child", "people": "person"},
    "link_iou": 0.3,                       # a detection belongs to a track when IoU >= this (link_track)
    "export_dir": "models/ovdetect",
    "export_dynamic": True,                # dynamic input shape: same letterbox as the PyTorch run
    "calib_dir": "data/calib_ov",
    "calib_frames": 300,                   # INT8 calibration frames (Ultralytics recommends >= 300)
    # calibration videos: every samples/uploads clip EXCEPT the demo clip the virtual cameras are cut from
    "calib_sources": ["samples/uploads/WhatsApp_Video_2026-10-07_at_15.59.14.mp4",
                      "samples/uploads/WhatsApp_Video_2026-10-08_at_15.57.23.mp4",
                      "samples/uploads/WhatsApp_Video_2026-10-09_at_07.21.29.mp4",
                      "samples/uploads/WhatsApp_Video_2026-10-09_at_07.21.30.mp4"],
    "calib_forbidden": ["samples/uploads/WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4"],
    # agreement check (PREREGISTRATION_v3 R3 / R5): 100 frames = 20 per video, boxes matched at IoU >= 0.5
    "agree_videos": ["data/virtual_cctv/videos/cam01.mp4", "data/virtual_cctv/videos/cam02.mp4",
                     "data/virtual_cctv/videos/cam03.mp4", "data/virtual_cctv/videos/cam04.mp4",
                     "samples/uploads/WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4"],
    "agree_frames_per_video": 20,
    "agree_iou": 0.5,
    "bench_frames": 20,                    # frames timed by `bench` (spread over the 4 virtual cameras)
    "bench_warmup": 2,
}
BACKENDS = ("yoloworld", "yoloe")
VOCAB_VERSION = 1
VOCAB_KEYS = ("imgsz", "conf", "nms_iou", "max_det", "every_s", "vocabulary")
DEMO_VIDEOS = ["data/virtual_cctv/videos/cam01.mp4", "data/virtual_cctv/videos/cam02.mp4",
               "data/virtual_cctv/videos/cam03.mp4", "data/virtual_cctv/videos/cam04.mp4"]
DISCLAIMER = ("Open-vocabulary detection (zero-shot detector, query text as the class name), not a verified "
              "identification: check the evidence image.")

_MODELS: dict[tuple, Any] = {}          # (backend, weights path) -> Ultralytics model
_CURRENT: dict[int, tuple] = {}         # id(model) -> class names currently set
_TEXT_PE: dict[tuple, Any] = {}         # (backend, weights path, classes) -> YOLOE text prompt embeddings
_TRACKS: dict[tuple, dict] = {}         # (stage1 path, stamp) -> track table (link_track)
_VIDEO_SIZE: dict[str, tuple] = {}      # video path -> (width, height)


def load_settings(overrides: Mapping[str, Any] | None = None, use_config: bool = True) -> dict:
    """OVDETECT_DEFAULTS, then config.yaml's optional `ovdetect:` section, then the non-None overrides.
    Dict-valued keys (backends, text_weights) are merged one level deep, so one backend can be overridden alone."""
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in OVDETECT_DEFAULTS.items()}

    def merge(extra: Mapping[str, Any]) -> None:
        for key, value in extra.items():
            if value is None:
                continue
            if isinstance(value, Mapping) and isinstance(settings.get(key), dict):
                settings[key].update(value)
            else:
                settings[key] = value

    if use_config:
        try:
            import utils
            merge(utils.load_config().get("ovdetect") or {})
        except Exception:                                     # a broken config must not stop the search
            pass
    merge(overrides or {})
    return settings


def _file_stamp(path: Path) -> str:
    """Size + mtime of a file ('' when missing): cheap change detection for the caches."""
    try:
        stat = Path(path).stat()
        return f"{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        return ""


def _weights_stamp(path: Path) -> str:
    """Change stamp of a model: the .pt file, or the .xml + .bin of an exported OpenVINO folder."""
    path = Path(path)
    if path.is_dir():
        return "|".join(f"{p.name}={_file_stamp(p)}" for p in sorted(path.glob("*.xml")) + sorted(path.glob("*.bin")))
    return _file_stamp(path)


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------- models (torch inside functions)

def _prepare_runtime(settings: Mapping[str, Any]) -> None:
    """Ultralytics settings folder, no auto-install / downloads, and the CPU thread cap. Must run before the first
    `import ultralytics` of the process to take full effect (environment variables are only set when unset)."""
    config_dir = camreg.resolve_path(settings.get("yolo_config_dir") or "../.ultralytics").resolve()
    if config_dir.is_dir():
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    import torch
    torch.set_num_threads(int(settings.get("threads") or 2))
    try:
        import cv2
        cv2.setNumThreads(int(settings.get("threads") or 2))
    except Exception:
        pass


def _is_exported(path: Path) -> bool:
    """True for an exported model (OpenVINO folder / .xml, ONNX), False for a PyTorch .pt checkpoint."""
    path = Path(path)
    return path.is_dir() or path.suffix.lower() in (".xml", ".onnx")


@contextlib.contextmanager
def openvino_threads(threads: int, device: str | None = "CPU"):
    """Inside the block every OpenVINO compile runs on `device` (CPU) with at most `threads` threads.
    Ultralytics' OpenVINO backend compiles with device AUTO (OpenVINO also lists this laptop's Radeon iGPU, so AUTO
    could move work there) and PERFORMANCE_HINT LATENCY = all cores; NNCF's INT8 calibration compiles with all cores."""
    try:
        import openvino as ov
    except ImportError:
        yield
        return
    original = ov.Core.compile_model

    def capped(self, model, device_name=None, config=None, **kwargs):
        config = dict(config or {})
        if device and (device_name is None or str(device_name).upper().startswith("AUTO")):
            device_name = device                              # AUTO -> CPU: the project runs on the CPU only
        if device_name is not None and str(device_name).upper().startswith("CPU"):
            asked = int(config.get("INFERENCE_NUM_THREADS") or 0)          # 0 = "all cores"
            config["INFERENCE_NUM_THREADS"] = min(asked, int(threads)) if asked > 0 else int(threads)
        return original(self, model, device_name, config, **kwargs)

    ov.Core.compile_model = capped
    try:
        yield
    finally:
        ov.Core.compile_model = original


def openvino_threads_in_use(model) -> int | None:
    """INFERENCE_NUM_THREADS of a loaded OpenVINO model (after its first predict), None for other formats."""
    try:
        return int(model.predictor.model.backend.ov_compiled_model.get_property("INFERENCE_NUM_THREADS"))
    except Exception:
        return None


def openvino_devices_in_use(model) -> list[str] | None:
    """EXECUTION_DEVICES of a loaded OpenVINO model (after its first predict), None for other formats."""
    try:
        return [str(d) for d in model.predictor.model.backend.ov_compiled_model.get_property("EXECUTION_DEVICES")]
    except Exception:
        return None


def _exported_names(path: Path) -> list[str] | None:
    """Class names baked into an exported model, read from its metadata.yaml (no compile needed); None if absent."""
    import yaml
    path = Path(path)
    meta_file = (path if path.is_dir() else path.parent) / "metadata.yaml"
    try:
        names = (yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}).get("names")
    except Exception:                                         # missing / unreadable metadata: caller falls back
        return None
    if isinstance(names, Mapping):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names] if names else None


def _seed_text_encoder(model, backend: str, settings: Mapping[str, Any]) -> None:
    """Give the model its text encoder from the project's weights folder (absolute path), so the working directory does
    not matter and Ultralytics never tries to download it."""
    import torch
    from ultralytics.nn import text_model as tm
    path = camreg.resolve_path(settings["text_weights"][backend]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{backend}: text encoder weights missing: {path} (not downloaded on purpose)")
    inner = model.model
    if backend == "yoloe":
        variant = str(getattr(inner, "text_model", ""))
        if not variant.startswith("mobileclip2"):
            raise RuntimeError(f"yoloe: text model {variant!r} needs other weights than {path.name}")
        inner.clip_model = tm.MobileCLIPTS(torch.device("cpu"), weight=str(path))
    else:
        old = tm.WEIGHTS_DIR
        tm.WEIGHTS_DIR = path.parent.parent                  # CLIP looks in <WEIGHTS_DIR>/clip/ViT-B-32.pt
        try:
            inner.clip_model = tm.build_text_model("clip:ViT-B/32", device=torch.device("cpu"))
        finally:
            tm.WEIGHTS_DIR = old


def _new_model(backend: str, path: Path, settings: Mapping[str, Any]):
    """A fresh (uncached) Ultralytics model: YOLOWorld / YOLOE for .pt, YOLO for exported folders."""
    _prepare_runtime(settings)
    from ultralytics import YOLO, YOLOE, YOLOWorld
    import cpu_limits
    cpu_limits.apply()      # set_classes() rebuilds the predictor, whose select_device() would reset torch to 7 threads
    if _is_exported(path):
        return YOLO(str(path))
    model = YOLOWorld(str(path)) if backend == "yoloworld" else YOLOE(str(path))
    _seed_text_encoder(model, backend, settings)
    return model


def _weights_path(backend: str, settings: Mapping[str, Any], weights: str | Path | None) -> Path:
    """Absolute path of the model to load for a backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r} (use {', '.join(BACKENDS)})")
    path = camreg.resolve_path(weights or settings["backends"][backend]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{backend}: model not found: {path}")
    return path


def load_model(backend: str, settings: Mapping[str, Any] | None = None, weights: str | Path | None = None):
    """The Ultralytics model of a backend (models/yolov8s-worldv2.pt or models/yoloe-26s-seg.pt, or `weights` = an
    exported model), loaded once per process on the CPU with `threads` threads."""
    settings = load_settings(settings, use_config=settings is None)
    path = _weights_path(backend, settings, weights)
    key = (backend, str(path))
    if key not in _MODELS:
        _MODELS[key] = _new_model(backend, path, settings)
    _prepare_runtime(settings)
    return _MODELS[key]


def _text_pe(model, backend: str, classes: tuple, settings: Mapping[str, Any]):
    """YOLOE text prompt embeddings of the class names (same as model.get_text_pe, but the text encoder stays loaded
    instead of being rebuilt on every call), cached per class list."""
    key = (backend, str(getattr(model.model, "pt_path", "")), classes)
    if key not in _TEXT_PE:
        import torch
        with torch.no_grad():
            _TEXT_PE[key] = model.model.get_text_pe(list(classes), cache_clip_model=True)
    return _TEXT_PE[key]


def set_classes(model, backend: str, classes: Iterable[str], settings: Mapping[str, Any] | None = None) -> None:
    """Make `classes` the detector's only classes (skipped when they are already set).
    YOLO-World: model.set_classes(names); YOLOE: model.set_classes(names, text embeddings)."""
    classes = tuple(str(c) for c in classes)
    if _CURRENT.get(id(model)) == classes:
        return
    if backend == "yoloworld":
        model.set_classes(list(classes))
    else:
        model.set_classes(list(classes), _text_pe(model, backend, classes, settings or OVDETECT_DEFAULTS))
    _CURRENT[id(model)] = classes


def _agnostic_nms(backend: str, settings: Mapping[str, Any]) -> bool:
    """Class-agnostic NMS for this backend (settings["agnostic_nms"] is a per-backend dict or one bool)."""
    value = settings.get("agnostic_nms", OVDETECT_DEFAULTS["agnostic_nms"])
    if isinstance(value, Mapping):
        value = value.get(backend, OVDETECT_DEFAULTS["agnostic_nms"].get(backend, False))
    return bool(value)


def detect(frames_bgr: list[np.ndarray], classes: Iterable[str], backend: str = "yoloe",
           settings: Mapping[str, Any] | None = None, weights: str | Path | None = None) -> list[list[dict]]:
    """Per frame (BGR, as read by cv2) a list of {"cls", "conf", "box": [x1, y1, x2, y2]} in the frame's own pixels,
    best first. An exported model has its vocabulary baked in: only its boxes of the requested classes are kept."""
    settings = load_settings(settings, use_config=settings is None)
    classes = [str(c) for c in classes]
    model = load_model(backend, settings, weights)
    exported = _is_exported(_weights_path(backend, settings, weights))
    def cap():
        """A fresh thread-cap block (a context manager object can be entered only once). An exported model is
        compiled lazily (first predict, or reading .names), so both always happen inside it."""
        if exported:
            return openvino_threads(int(settings["threads"]), settings.get("openvino_device"))
        return contextlib.nullcontext()

    if exported:
        names = _exported_names(_weights_path(backend, settings, weights))
        if names is None:                                     # no metadata.yaml: ask the model (compiles it)
            with cap():
                names = model.names
        baked = set(names.values()) if isinstance(names, Mapping) else set(names)
        missing = [c for c in classes if c not in baked]
        if missing:
            raise ValueError(f"exported model has no class {missing[0]!r} (its vocabulary is fixed)")
    else:
        set_classes(model, backend, classes, settings)
    wanted = set(classes)
    agnostic = _agnostic_nms(backend, settings)
    out = []
    for frame in frames_bgr:
        with cap():
            result = model.predict(frame, imgsz=int(settings["imgsz"]), conf=float(settings["conf"]),
                                   iou=float(settings["nms_iou"]), max_det=int(settings["max_det"]),
                                   agnostic_nms=agnostic, device="cpu", verbose=False)[0]
        boxes = result.boxes
        dets = []
        if boxes is not None and len(boxes):
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                name = result.names[int(cls)]
                if name in wanted:
                    dets.append({"cls": name, "conf": round(float(conf), 4), "box": [round(float(v), 1) for v in xyxy]})
        out.append(sorted(dets, key=lambda d: -d["conf"]))
    return out


# ---------------------------------------------------------------------------- video frames and tracks

def read_frames(video: str | Path, frame_numbers: Iterable[int]) -> dict[int, np.ndarray]:
    """{frame number: BGR frame} read by seeking (checked frame-exact on the demo videos); unreadable frames are
    left out."""
    import cv2
    frames = {}
    cap = cv2.VideoCapture(str(camreg.resolve_path(video)))
    try:
        if not cap.isOpened():
            return frames
        for number in sorted(set(int(n) for n in frame_numbers)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = cap.read()
            if ok:
                frames[number] = frame
    finally:
        cap.release()
    return frames


def _video_size(video: str | Path) -> tuple[int, int] | None:
    """(width, height) of a video, cached; None when it cannot be opened."""
    key = str(camreg.resolve_path(video))
    if key not in _VIDEO_SIZE:
        import cv2
        cap = cv2.VideoCapture(key)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if size[0] <= 0 or size[1] <= 0:
            return None
        _VIDEO_SIZE[key] = size
    return _VIDEO_SIZE[key]


def _track_table(camera: Mapping[str, Any]) -> dict:
    """{"stride", "proc_size", "by_frame": {frame: [(track id, box in processing pixels)]}} of a camera's
    stage1_tracking.jsonl, cached per file version. Empty table when the file is missing."""
    path = camreg.resolve_path(camera.get("out_dir") or "") / "stage1_tracking.jsonl"
    key = (str(path), _file_stamp(path))
    if key not in _TRACKS:
        table = {"stride": 1, "proc_size": None, "by_frame": {}}
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if "meta" in row:
                        table["stride"] = max(1, int(row["meta"].get("stride") or 1))
                        table["proc_size"] = row["meta"].get("proc_size")
                    elif row.get("person_id") is not None and row.get("bbox"):
                        table["by_frame"].setdefault(int(row["frame"]), []).append(
                            (int(row["person_id"]), [float(v) for v in row["bbox"]]))
        _TRACKS[key] = table
    return _TRACKS[key]


def link_track(camera: Mapping[str, Any], frame: int, box: Iterable[float], iou_thr: float | None = None,
               video_size: tuple[int, int] | None = None, settings: Mapping[str, Any] | None = None) -> int | None:
    """The local track id whose box (from stage1_tracking.jsonl, scaled from processing to video pixels) overlaps the
    detection `box` (video pixels) best with IoU >= iou_thr, or None. Each track uses its row with the nearest frame
    number within the tracker stride. `video_size` = (w, h) of the frame (read from the video when None)."""
    if iou_thr is None:
        iou_thr = float((settings or OVDETECT_DEFAULTS).get("link_iou", OVDETECT_DEFAULTS["link_iou"]))
    table = _track_table(camera)
    if not table["by_frame"]:
        return None
    if video_size is None:
        video_size = _video_size(camera.get("video") or "")
        if video_size is None:
            return None
    proc = table["proc_size"] or list(video_size)
    sx, sy = video_size[0] / float(proc[0]), video_size[1] / float(proc[1])
    nearest: dict[int, list[float]] = {}                 # track id -> its box at the nearest frame
    frame = int(frame)
    for distance in range(table["stride"] + 1):
        for number in ((frame,) if distance == 0 else (frame - distance, frame + distance)):
            for track_id, b in table["by_frame"].get(number, []):
                nearest.setdefault(track_id, [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy])
    best_id, best_iou = None, float(iou_thr)
    for track_id, tbox in nearest.items():
        iou = box_iou(box, tbox)
        if iou >= best_iou and (best_id is None or iou > best_iou):
            best_id, best_iou = track_id, iou
    return best_id


# ---------------------------------------------------------------------------- grounding (retrieve, then detect)

def _camera_map(cameras: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    """{camera_id: camera row} from a mapping or a list of rows (EventIndex.list_cameras())."""
    if not cameras:
        return {}
    if isinstance(cameras, Mapping):
        return dict(cameras)
    return {row["camera_id"]: row for row in cameras}


def rank_detections(candidates: list[dict], per_frame: list[list[dict]], query: str, threshold: float,
                    backend: str) -> list[dict]:
    """One hit per candidate frame = its best box with conf >= threshold; ranked by detector confidence, ties by CLIP
    rank. Frames without such a box are dropped. NumPy-free, so the tests check it directly."""
    hits = []
    for cand, dets in zip(candidates, per_frame):
        good = [d for d in dets if d["cls"] == query and float(d["conf"]) >= threshold]
        if not good:
            continue
        best = max(good, key=lambda d: float(d["conf"]))
        hits.append({"kind": "detection", "camera_id": cand["camera_id"], "track_id": None,
                     "score": float(best["conf"]), "t": cand["t"], "frame": cand["frame"],
                     "box": [float(v) for v in best["box"]], "span": [cand["t"], cand["t"]],
                     "clip_score": cand["clip_score"], "clip_rank": cand["clip_rank"], "backend": backend,
                     "cls": query, "n_boxes": len(good)})
    return sorted(hits, key=lambda h: (-h["score"], h["clip_rank"]))


def ground(index: Mapping[str, Any], query: str, cameras: Mapping[str, Any] | Iterable[Mapping[str, Any]],
           settings: Mapping[str, Any] | None = None, backend: str = "yoloe", cams: Iterable[str] | None = None,
           window: tuple | Mapping | None = None, top_k: int | None = None,
           detect_fn: Callable[[list, list], list] | None = None, text_fn=None, weights: str | Path | None = None,
           frame_fn: Callable[[Mapping[str, Any], list], dict] | None = None, stats: dict | None = None,
           echo=print) -> list[dict]:
    """Retrieve, then detect. Stage 1: the CLIP frame baseline (openvocab.search, method "baseline_frame") gives the
    K = `candidates_k` best frames in scope. Stage 2: the detector with the query text as its only class on those
    frames; one hit per frame with a box >= `conf` (hit format of openvocab plus clip_score, clip_rank, backend).
    Returns [] when no candidate frame has such a box ("no confident match").

    detect_fn(frames, classes) / text_fn(texts) / frame_fn(camera, frame numbers) replace the detector, the CLIP text
    encoder and the video reader in tests. `stats` (a dict) receives the time of each stage in ms."""
    import openvocab
    settings = load_settings(settings, use_config=settings is None)
    top_k = int(top_k or settings["top_k"])
    threshold = float(settings["conf"])
    cam_rows = _camera_map(cameras)
    timing = stats if stats is not None else {}
    if index is None or not str(query or "").strip():
        return []
    started = time.perf_counter()
    stage1 = openvocab.search(index, query, openvocab.load_settings(), method="baseline_frame", cams=cams,
                              window=window, top_k=int(settings["candidates_k"]), text_fn=text_fn)
    timing["stage1_ms"] = round((time.perf_counter() - started) * 1000, 1)
    candidates = [{"camera_id": h["camera_id"], "frame": int(h["frame"]), "t": float(h["t"]),
                   "clip_score": float(h["score"]), "clip_rank": rank} for rank, h in enumerate(stage1, start=1)]
    started = time.perf_counter()
    images: dict[tuple, np.ndarray] = {}
    for cam_id in sorted({c["camera_id"] for c in candidates}):
        camera = cam_rows.get(cam_id)
        if camera is None:
            echo(f"  ovdetect: camera {cam_id} not in the camera list, its candidates are skipped")
            continue
        numbers = [c["frame"] for c in candidates if c["camera_id"] == cam_id]
        try:
            got = (frame_fn or (lambda cam, nums: read_frames(cam.get("video") or "", nums)))(camera, numbers)
        except Exception as exc:                              # an unreadable video must not stop the answer
            echo(f"  ovdetect: cannot read {cam_id}: {exc}")
            got = {}
        for number, image in got.items():
            images[(cam_id, int(number))] = image
    readable = [c for c in candidates if (c["camera_id"], c["frame"]) in images]
    if len(readable) < len(candidates):
        echo(f"  ovdetect: {len(candidates) - len(readable)} candidate frame(s) could not be read")
    timing["read_ms"] = round((time.perf_counter() - started) * 1000, 1)
    frames = [images[(c["camera_id"], c["frame"])] for c in readable]
    if detect_fn is None and frames and not _is_exported(_weights_path(backend, settings, weights)):
        started = time.perf_counter()                     # query text -> detector class (text encoder), per query
        set_classes(load_model(backend, settings, weights), backend, [query], settings)
        timing["classes_ms"] = round((time.perf_counter() - started) * 1000, 1)
    started = time.perf_counter()
    if detect_fn is None:
        per_frame = detect(frames, [query], backend, settings, weights) if frames else []
    else:
        per_frame = detect_fn(frames, [query]) if frames else []
    timing["detect_ms"] = round((time.perf_counter() - started) * 1000, 1)
    timing["candidates"] = len(candidates)
    hits = rank_detections(readable, per_frame, query, threshold, backend)[:top_k]
    for hit in hits:
        camera = cam_rows.get(hit["camera_id"]) or {}
        image = images.get((hit["camera_id"], hit["frame"]))
        size = (image.shape[1], image.shape[0]) if image is not None else None
        try:
            hit["track_id"] = link_track(camera, hit["frame"], hit["box"], video_size=size, settings=settings)
        except Exception:                                     # no tracking file: the hit stays without a track
            hit["track_id"] = None
    return hits


# ---------------------------------------------------------------------------- fixed vocabulary

def vocab_term(query: str, settings: Mapping[str, Any] | None = None) -> str | None:
    """The vocabulary term the WHOLE query names ('Any life jackets?' -> 'life jacket'), else None. Lower case,
    one leading a/an/the/any/some removed, plural (+s / +es, children, people) accepted."""
    settings = load_settings(settings, use_config=settings is None)
    vocab = {str(v).lower() for v in settings["vocabulary"]}
    text = re.sub(r"\s+", " ", str(query or "").lower()).strip().strip("?!.,;: ")
    text = re.sub(r"^(?:a|an|the|any|some)\s+", "", text).strip()
    if text in vocab:
        return text
    irregular = {str(k).lower(): str(v).lower() for k, v in (settings.get("irregular_plurals") or {}).items()}
    if irregular.get(text) in vocab:
        return irregular[text]
    for suffix in ("s", "es"):
        if text.endswith(suffix) and text[:-len(suffix)] in vocab:
            return text[:-len(suffix)]
    return None


def vocab_index_path(db_path: str | Path, backend: str) -> Path:
    """<db dir>/ovdetect_vocab_<backend>.json."""
    return Path(db_path).parent / f"ovdetect_vocab_{backend}.json"


def _read_json(path: str | Path) -> dict | None:
    """A JSON file, or None when missing / unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_vocab_index(path: str | Path) -> dict | None:
    """The saved vocabulary index, or None when missing / unreadable."""
    return _read_json(path)


def _scan_camera(camera: Mapping[str, Any], classes: list[str], every_s: float, detect_one: Callable,
                 echo=print) -> dict:
    """Detections of one camera on one frame every `every_s` seconds (frame step as in openvocab.build)."""
    import cv2
    cam_id = camera["camera_id"]
    video = camreg.resolve_path(camera.get("video") or "")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"{cam_id}: cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(every_s * fps)))
    detections, sampled, index, size = [], [], 0, None
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                size = [frame.shape[1], frame.shape[0]]
                t = round(index / fps, 3)
                sampled.append(index)
                for det in detect_one(frame, classes):
                    detections.append({"camera_id": cam_id, "frame": index, "t": t, **det})
            index += 1
    finally:
        cap.release()
    seconds = time.perf_counter() - started
    echo(f"  {cam_id}: {len(sampled)} frames, {len(detections)} boxes, {seconds:.1f} s")
    return {"detections": detections, "sampled": sampled, "video_size": size, "fps": fps, "seconds": round(seconds, 1),
            "out_dir": camera.get("out_dir"), "video": camera.get("video")}


def build_vocab_index(db_path: str | Path, backend: str, settings: Mapping[str, Any] | None = None,
                      weights: str | Path | None = None, force: bool = False, echo=print,
                      detect_fn: Callable[[list, list], list] | None = None) -> Path:
    """Run the detector with the whole fixed vocabulary on every camera of the event index every `every_s` seconds and
    save <db dir>/ovdetect_vocab_<backend>.json. Cameras whose video and model are unchanged are reused."""
    from event_index import EventIndex
    settings = load_settings(settings, use_config=settings is None)
    out = vocab_index_path(db_path, backend)
    old = None if force else load_vocab_index(out)
    model_path = _weights_path(backend, settings, weights) if detect_fn is None else Path(str(weights or "fake"))
    keyed = {k: settings[k] for k in VOCAB_KEYS}
    keyed["weights"] = model_path.name
    keyed["agnostic_nms"] = _agnostic_nms(backend, settings)
    if old and "agnostic_nms" not in (old.get("settings") or {}):
        # files written before this key existed were built with Ultralytics' own default for the backend
        old["settings"] = {**(old.get("settings") or {}), "agnostic_nms":OVDETECT_DEFAULTS["agnostic_nms"].get(backend, False)}
    classes = list(settings["vocabulary"])

    def detect_one(frame, names):
        if detect_fn is not None:
            return detect_fn([frame], names)[0]
        return detect([frame], names, backend, settings, weights)[0]

    cameras = {}
    started = time.perf_counter()
    for cam in EventIndex(db_path).list_cameras():
        cam_id = cam["camera_id"]
        stamp = {"video": _file_stamp(camreg.resolve_path(cam.get("video") or "")),
                 "model": _weights_stamp(model_path)}
        reuse = (old and old.get("version") == VOCAB_VERSION and old.get("settings") == keyed
                 and (old.get("cameras") or {}).get(cam_id, {}).get("stamp") == stamp)
        if reuse:
            echo(f"  {cam_id}: unchanged, reused")
            part = dict(old["cameras"][cam_id])
            part["detections"] = [d for d in old["detections"] if d["camera_id"] == cam_id]
        else:
            part = _scan_camera(cam, classes, float(settings["every_s"]), detect_one, echo=echo)
        part["stamp"] = stamp
        cameras[cam_id] = part
    detections = [d for part in cameras.values() for d in part.pop("detections")]
    info = {"version": VOCAB_VERSION, "backend": backend, "settings": keyed, "db": Path(db_path).name,
            "built_at": datetime.now().isoformat(timespec="seconds"), "cameras": cameras, "detections": detections,
            "build_s": round(time.perf_counter() - started, 1)}
    out.write_text(json.dumps(info), encoding="utf-8")
    echo(f"vocabulary index ({backend}): {sum(len(p['sampled']) for p in cameras.values())} frames, "
         f"{len(detections)} boxes -> {out}")
    return out


def vocab_search(vocab_index: Mapping[str, Any] | str | Path, term: str, cams: Iterable[str] | None = None,
                 window: tuple | Mapping | None = None, top_k: int | None = 5,
                 settings: Mapping[str, Any] | None = None, link: bool = True) -> list[dict]:
    """Hits (kind "vocab") for one vocabulary term from the saved index: the best box per camera per
    `same_moment_s` seconds, highest confidence first. track_id is linked from the camera's tracking file."""
    import openvocab
    settings = load_settings(settings, use_config=settings is None)
    if not isinstance(vocab_index, Mapping):
        vocab_index = load_vocab_index(vocab_index) or {}
    term = str(term).lower()
    cams = list(cams) if cams else None
    same = float(settings.get("same_moment_s", OVDETECT_DEFAULTS["same_moment_s"]))
    found = [d for d in vocab_index.get("detections", [])
             if str(d["cls"]).lower() == term and openvocab._in_scope(d, cams, window)]
    hits: list[dict] = []
    for det in sorted(found, key=lambda d: (-float(d["conf"]), float(d["t"]), d["camera_id"])):
        if any(h["camera_id"] == det["camera_id"] and abs(h["t"] - float(det["t"])) < same for h in hits):
            continue
        hits.append({"kind": "vocab", "camera_id": det["camera_id"], "track_id": None, "score": float(det["conf"]),
                     "t": float(det["t"]), "frame": int(det["frame"]), "box": [float(v) for v in det["box"]],
                     "span": [float(det["t"]), float(det["t"])], "cls": term,
                     "backend": vocab_index.get("backend")})
        if top_k and len(hits) >= int(top_k):
            break
    if link:
        for hit in hits:
            part = (vocab_index.get("cameras") or {}).get(hit["camera_id"]) or {}
            size = part.get("video_size")
            try:
                hit["track_id"] = link_track({"out_dir": part.get("out_dir"), "video": part.get("video")},
                                             hit["frame"], hit["box"], video_size=tuple(size) if size else None,
                                             settings=settings)
            except Exception:
                hit["track_id"] = None
    return hits


# ---------------------------------------------------------------------------- export (OpenVINO, INT8)

def prepare_calibration(settings: Mapping[str, Any] | None = None, echo=print) -> Path:
    """INT8 calibration frames in data/calib_ov/images (evenly spread over the `calib_sources` videos, never the demo
    clip) + data/calib_ov/calib.yaml for the Ultralytics exporter. Reused when the manifest matches."""
    import cv2
    import yaml
    settings = load_settings(settings, use_config=settings is None)
    calib_dir = camreg.resolve_path(settings["calib_dir"])
    images = calib_dir / "images"
    forbidden = {camreg.resolve_path(p).resolve() for p in settings["calib_forbidden"]}
    sources = [camreg.resolve_path(p).resolve() for p in settings["calib_sources"]]
    for src in sources:
        if src in forbidden or src.name in {f.name for f in forbidden}:
            raise ValueError(f"calibration source {src.name} is the demo clip: not allowed")
    total = int(settings["calib_frames"])
    manifest_path = calib_dir / "manifest.json"
    wanted = {"sources": [p.relative_to(ROOT).as_posix() for p in sources], "frames": total,
              "stamps": [_file_stamp(p) for p in sources]}
    old = _read_json(manifest_path)
    if old and {k: old.get(k) for k in wanted} == wanted and all((images / f).is_file() for f in old["files"]):
        echo(f"calibration: {len(old['files'])} frames reused from {images}")
        files = old["files"]
    else:
        images.mkdir(parents=True, exist_ok=True)
        for stale in images.glob("*.jpg"):
            stale.unlink()
        files = []
        per_video = [total // len(sources) + (1 if i < total % len(sources) else 0) for i in range(len(sources))]
        for src, count in zip(sources, per_video):
            cap = cv2.VideoCapture(str(src))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            picks = set(np.linspace(0, max(n - 1, 0), count).round().astype(int).tolist())
            index = 0
            while picks:
                ok, frame = cap.read()
                if not ok:
                    break
                if index in picks:
                    name = f"{src.stem}_{index:05d}.jpg"
                    cv2.imwrite(str(images / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    files.append(name)
                    picks.discard(index)
                index += 1
            cap.release()
        manifest_path.write_text(json.dumps({**wanted, "files": files,
                                             "made_at": datetime.now().isoformat(timespec="seconds")}, indent=1),
                                 encoding="utf-8")
        echo(f"calibration: {len(files)} frames written to {images}")
    data_yaml = calib_dir / "calib.yaml"
    names = {i: name for i, name in enumerate(settings["vocabulary"])}
    # Ultralytics' check_det_dataset requires both 'train' and 'val'; only 'val' is read for INT8 calibration.
    # No absolute 'path': Ultralytics then resolves the folders next to this file, so moving the project keeps it valid.
    data_yaml.write_text(yaml.safe_dump({"train": "images", "val": "images", "names": names}, sort_keys=False),
                         encoding="utf-8")
    return data_yaml


def export_vocab_model(backend: str, fmt: str = "openvino", int8: bool = True,
                       settings: Mapping[str, Any] | None = None, out_dir: str | Path | None = None,
                       calib_max: int | None = None, echo=print) -> Path:
    """Export the detector with the fixed vocabulary set: set_classes(vocabulary) on a fresh FP32 model BEFORE the
    export (YOLO-World keeps its text features, YOLOE fuses its text embeddings into the head), then Ultralytics'
    exporter -> models/ovdetect/<backend>_vocab_[int8_]openvino_model/. NMS mode = Ultralytics' default (nms=None:
    one-to-many head + NMS), the same as the PyTorch predictor. INT8 is calibrated on frames of other videos than the
    demo clip (prepare_calibration). Raises when the export fails or lost its vocabulary; returns the exported path."""
    settings = load_settings(settings, use_config=settings is None)
    out_dir = Path(out_dir) if out_dir else camreg.resolve_path(settings["export_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab = list(settings["vocabulary"])
    data_yaml = prepare_calibration(settings, echo=echo) if int8 else None
    started = time.perf_counter()
    model = _new_model(backend, _weights_path(backend, settings, None), settings)    # not the cached live model
    set_classes(model, backend, vocab, settings)
    _CURRENT.pop(id(model), None)
    model.model.clip_model = None                        # the text encoder is not part of the exported graph
    # the exporter names its output after pt_path: models/ovdetect/<backend>_vocab_[int8_]openvino_model/
    model.model.pt_path = str(out_dir / f"{backend}_vocab.pt")
    n_calib = 0
    fraction = 1.0
    if int8:
        n_calib = len(list((Path(data_yaml).parent / "images").glob("*.jpg")))
        if calib_max:
            fraction = min(1.0, int(calib_max) / max(n_calib, 1))
    import ultralytics.engine.exporter as exporter_module
    real_settings = exporter_module.SETTINGS
    exporter_module.SETTINGS = dict(real_settings)       # the exporter would rewrite settings.json for a tip flag
    try:
        with openvino_threads(int(settings["threads"]), settings.get("openvino_device")):    # NNCF calibration too
            exported = model.export(format=fmt, imgsz=int(settings["imgsz"]), quantize=8 if int8 else None,
                                    data=str(data_yaml) if int8 else None, dynamic=bool(settings["export_dynamic"]),
                                    fraction=fraction, batch=1, device="cpu", verbose=False)
    finally:
        exporter_module.SETTINGS = real_settings
    exported = Path(str(exported))
    seconds = round(time.perf_counter() - started, 1)
    names = _exported_names(exported)
    if names != vocab:
        raise RuntimeError(f"{backend}: the exported model lost its vocabulary ({names!r:.80})")
    info = {"backend": backend, "format": fmt, "int8": bool(int8), "vocabulary": vocab,
            "source_weights": settings["backends"][backend], "imgsz": int(settings["imgsz"]),
            "dynamic": bool(settings["export_dynamic"]), "nms": "Ultralytics default (None), as the PyTorch predictor",
            "export_s": seconds, "names_checked": "metadata.yaml names == vocabulary",
            "calibration": ({"yaml": str(Path(data_yaml).relative_to(ROOT).as_posix()), "images_available": n_calib,
                             "fraction": round(fraction, 4), "calib_max": calib_max,
                             "sources": settings["calib_sources"]} if int8 else None),
            "made_at": datetime.now().isoformat(timespec="seconds")}
    sidecar = (exported if exported.is_dir() else exported.parent) / "ovdetect_export.json"
    sidecar.write_text(json.dumps(info, indent=1), encoding="utf-8")
    echo(f"exported {backend} ({fmt}{', INT8' if int8 else ''}) in {seconds:.1f} s -> {exported}")
    return exported


# ---------------------------------------------------------------------------- checks and measurements

def _even_frames(videos: Iterable[str], per_video: int) -> list[tuple[str, int, np.ndarray]]:
    """(video, frame number, BGR frame): `per_video` frames spread evenly over each video."""
    import cv2
    out = []
    for video in videos:
        cap = cv2.VideoCapture(str(camreg.resolve_path(video)))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        numbers = np.linspace(0, max(n - 1, 0), per_video).round().astype(int).tolist()
        got = read_frames(video, numbers)
        out += [(video, number, got[number]) for number in numbers if number in got]
    return out


def match_f1(reference: list[list[dict]], candidate: list[list[dict]], iou_thr: float) -> dict:
    """Agreement of two detectors on the same frames: boxes of the same class matched one-to-one (greedy, highest
    IoU first) at IoU >= iou_thr; F1 of the candidate against the reference. Also the confidences of the unmatched
    boxes (fp = only in the candidate, fn = only in the reference), to see whether they sit at the threshold."""
    tp = fp = fn = 0
    fp_conf, fn_conf, matched = [], [], []
    for ref, cand in zip(reference, candidate):
        pairs = sorted(((box_iou(r["box"], c["box"]), i, j) for i, r in enumerate(ref) for j, c in enumerate(cand)
                        if r["cls"] == c["cls"]), reverse=True)
        used_r, used_c = set(), set()
        for iou, i, j in pairs:
            if iou < iou_thr:
                break
            if i not in used_r and j not in used_c:
                used_r.add(i)
                used_c.add(j)
                matched.append((iou, abs(float(ref[i]["conf"]) - float(cand[j]["conf"]))))
        tp += len(used_r)
        fn += len(ref) - len(used_r)
        fp += len(cand) - len(used_c)
        fn_conf += [round(float(r["conf"]), 3) for i, r in enumerate(ref) if i not in used_r]
        fp_conf += [round(float(c["conf"]), 3) for j, c in enumerate(cand) if j not in used_c]
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4), "fp_conf": sorted(fp_conf), "fn_conf": sorted(fn_conf),
            "matched_iou_mean": round(float(np.mean([m[0] for m in matched])), 4) if matched else None,
            "matched_conf_absdiff_mean": round(float(np.mean([m[1] for m in matched])), 4) if matched else None,
            "matched_conf_absdiff_max": round(float(np.max([m[1] for m in matched])), 4) if matched else None}


def agreement(backend: str, weights: str | Path, settings: Mapping[str, Any] | None = None,
              per_video: int | None = None, echo=print) -> dict:
    """PREREGISTRATION_v3 R5: exported vocabulary model vs the PyTorch FP32 model with the same vocabulary on
    `agree_frames_per_video` frames of each `agree_videos` video (conf >= `conf`, same class, IoU >= `agree_iou`)."""
    settings = load_settings(settings, use_config=settings is None)
    per_video = int(per_video or settings["agree_frames_per_video"])
    frames = _even_frames(settings["agree_videos"], per_video)
    images = [f for _, _, f in frames]
    vocab = list(settings["vocabulary"])
    result = {"backend": backend, "candidate": str(weights), "frames": len(images)}
    for label, w in (("reference", None), ("candidate", weights)):
        detect([images[0]], vocab, backend, settings, w)                  # load + warm up before timing
        started = time.perf_counter()
        result[label] = detect(images, vocab, backend, settings, w)
        result[f"{label}_ms_per_frame"] = round((time.perf_counter() - started) * 1000 / max(len(images), 1), 1)
        echo(f"  {label}: {sum(len(d) for d in result[label])} boxes, {result[f'{label}_ms_per_frame']} ms/frame")
    result["agreement"] = match_f1(result.pop("reference"), result.pop("candidate"), float(settings["agree_iou"]))
    exported_model = load_model(backend, settings, weights)
    result["openvino_threads"] = openvino_threads_in_use(exported_model)
    result["openvino_devices"] = openvino_devices_in_use(exported_model)
    result["rule"] = f"R5: adopt INT8 only if F1 >= 0.98 (boxes matched at IoU >= {settings['agree_iou']}, same class)"
    return result


def bench(backend: str, settings: Mapping[str, Any] | None = None, weights: str | Path | None = None,
          classes: list[str] | None = None, n_frames: int | None = None, echo=print) -> dict:
    """Detector milliseconds per frame (one predict call per frame, after `bench_warmup` warm-up frames) on frames
    spread over the 4 virtual cameras."""
    settings = load_settings(settings, use_config=settings is None)
    n_frames = int(n_frames or settings["bench_frames"])
    classes = list(classes or ["person"])
    per_video = max(1, math.ceil(n_frames / len(DEMO_VIDEOS)))
    frames = [f for _, _, f in _even_frames(DEMO_VIDEOS, per_video)][:n_frames]
    started = time.perf_counter()
    model = load_model(backend, settings, weights)
    load_s = time.perf_counter() - started
    for frame in frames[:int(settings["bench_warmup"])]:
        detect([frame], classes, backend, settings, weights)
    times = []
    for frame in frames:
        started = time.perf_counter()
        detect([frame], classes, backend, settings, weights)
        times.append((time.perf_counter() - started) * 1000)
    result = {"backend": backend, "weights": str(weights or settings["backends"][backend]), "classes": len(classes),
              "frames": len(times), "imgsz": int(settings["imgsz"]), "threads": int(settings["threads"]),
              "load_s": round(load_s, 2), "ms_mean": round(float(np.mean(times)), 1),
              "ms_median": round(float(np.median(times)), 1), "ms_min": round(float(np.min(times)), 1),
              "ms_max": round(float(np.max(times)), 1), "openvino_threads": openvino_threads_in_use(model),
              "openvino_devices": openvino_devices_in_use(model)}
    echo(json.dumps(result))
    return result


def colour_check(backend: str, settings: Mapping[str, Any] | None = None, video: str = DEMO_VIDEOS[3],
                 frame_number: int = 0) -> dict:
    """Which channel order reaches the network, and does a channel-swapped frame really reach it swapped?

    The hook sits on the first 3-channel conv of the model the PREDICTOR runs (AutoBackend may run a deep copy of
    model.model, and a hook on the original would then not see - or, if it returns a value, would even replace - the
    input). It only copies the input (returns None). The captured tensor is compared with the letterboxed BGR frame
    (channel 0 = R means the predictor flipped BGR to RGB, as in training). For a segmentation model (YOLOE) the
    plot is also checked: result.plot() only changes pixels inside the detected boxes (mask + box overlay)."""
    import torch
    from ultralytics.data.augment import LetterBox
    settings = load_settings(settings, use_config=settings is None)
    frame = read_frames(video, [frame_number])[frame_number]
    swapped_frame = np.ascontiguousarray(frame[..., ::-1])
    model = load_model(backend, settings)
    detect([frame], ["person"], backend, settings)                      # builds the predictor for these classes
    backend_obj = getattr(model.predictor.model, "backend", None)        # AutoBackend -> PyTorchBackend -> model
    net = getattr(backend_obj, "model", None) or model.predictor.model
    convs = [m for m in net.modules() if isinstance(m, torch.nn.Conv2d) and m.in_channels == 3]
    seen: list = []
    hook = convs[0].register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))   # returns None
    try:
        normal = detect([frame], ["person"], backend, settings)[0]
        swapped = detect([swapped_frame], ["person"], backend, settings)[0]
    finally:
        hook.remove()
    x = seen[0][0].float().numpy() * 255.0
    xs = seen[1][0].float().numpy() * 255.0
    letterboxed = LetterBox(tuple(x.shape[1:]), auto=False, stride=32)(image=frame).astype(np.float32)
    as_rgb = float(np.abs(x[0] - letterboxed[..., 2]).mean() + np.abs(x[2] - letterboxed[..., 0]).mean())
    as_bgr = float(np.abs(x[0] - letterboxed[..., 0]).mean() + np.abs(x[2] - letterboxed[..., 2]).mean())
    result = {"backend": backend, "video": video, "frame": frame_number, "input_shape": list(seen[0].shape),
              "diff_if_rgb": round(as_rgb, 3), "diff_if_bgr": round(as_bgr, 3),
              "network_receives": "RGB" if as_rgb < as_bgr else "BGR",
              # the swapped frame reached the network swapped: its channel 0 equals the normal channel 2
              "swapped_tensor_mean_abs_diff": round(float(np.abs(x - xs).mean()), 3),
              "swapped_ch0_vs_normal_ch2": round(float(np.abs(xs[0] - x[2]).mean()), 3),
              "predictor_runs_a_copy": not any(m is convs[0] for m in model.model.modules()),
              "boxes_bgr_input": [[d["conf"], [round(v) for v in d["box"]]] for d in normal],
              "boxes_swapped_input": [[d["conf"], [round(v) for v in d["box"]]] for d in swapped]}
    raw = model.predict(frame, imgsz=int(settings["imgsz"]), conf=float(settings["conf"]), device="cpu",
                        verbose=False)[0]
    plotted = raw.plot(labels=False, conf=False).astype(np.int16)      # BGR, as cv2.imwrite expects; no text
    changed = np.abs(plotted - frame.astype(np.int16)).sum(axis=2) > 0
    inside = np.zeros(changed.shape, bool)
    for x1, y1, x2, y2 in raw.boxes.xyxy.tolist() if raw.boxes is not None else []:
        inside[max(0, int(y1) - 3):int(y2) + 3, max(0, int(x1) - 3):int(x2) + 3] = True     # + box line width
    result["plot"] = {"has_masks": raw.masks is not None, "frame_pixels": int(changed.size),
                      "changed_pixels": int(changed.sum()),
                      "changed_outside_boxes": int((changed & ~inside).sum()),
                      "plot_vs_swapped_frame_mean_abs_diff": round(float(
                          np.abs(plotted - swapped_frame.astype(np.int16))[~inside].mean()), 3) if (~inside).any()
                          else None}
    return result


# ---------------------------------------------------------------------------- CLI

def lower_own_priority() -> str:
    """Put this process at below-normal priority (Windows) / nice +5 (elsewhere); returns what was done."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            ok = kernel32.SetPriorityClass(ctypes.c_void_p(kernel32.GetCurrentProcess()), 0x00004000)  # BELOW_NORMAL
            return "below normal" if ok else "unchanged (SetPriorityClass failed)"
        os.nice(5)
        return "nice +5"
    except Exception as exc:                                  # never stop the run for this
        return f"unchanged ({type(exc).__name__})"


def _cameras_of(db: str) -> dict[str, dict]:
    """{camera_id: camera row} of an event index."""
    from event_index import EventIndex
    return {row["camera_id"]: row for row in EventIndex(db).list_cameras()}


def main(argv=None) -> int:
    """ground / vocab / vocab-search / export / bench / agree / check-colour / calib from the command line."""
    parser = argparse.ArgumentParser(description="AquaWatch open-vocabulary detection (YOLO-World / YOLOE)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("ground", help="retrieve (CLIP frames) then detect (query text as the class)")
    g.add_argument("--db", required=True)
    g.add_argument("--backend", default="yoloe", choices=BACKENDS)
    g.add_argument("--weights", default=None, help="another model file (default: the backend's .pt)")
    g.add_argument("--evidence", action="store_true", help="write an evidence JPG per hit")
    g.add_argument("text", nargs="+")
    v = sub.add_parser("vocab", help="build the fixed-vocabulary index of every camera")
    v.add_argument("--db", required=True)
    v.add_argument("--backend", default="yoloe", choices=BACKENDS)
    v.add_argument("--weights", default=None, help="e.g. an exported vocabulary model")
    v.add_argument("--force", action="store_true")
    vs = sub.add_parser("vocab-search", help="search the fixed-vocabulary index for one term")
    vs.add_argument("--db", required=True)
    vs.add_argument("--backend", default="yoloe", choices=BACKENDS)
    vs.add_argument("text", nargs="+")
    e = sub.add_parser("export", help="export the vocabulary model (OpenVINO, optional INT8)")
    e.add_argument("--backend", default="yoloe", choices=BACKENDS)
    e.add_argument("--format", default="openvino")
    e.add_argument("--int8", action="store_true")
    e.add_argument("--out-dir", default=None)
    e.add_argument("--calib-max", type=int, default=None, help="use at most N calibration frames (dev runs)")
    b = sub.add_parser("bench", help="detector ms per frame")
    b.add_argument("--backend", default="yoloe", choices=BACKENDS)
    b.add_argument("--weights", default=None)
    b.add_argument("--frames", type=int, default=None)
    b.add_argument("--vocab", action="store_true", help="time with the whole vocabulary as classes")
    b.add_argument("--out", default=None, help="also write the result as JSON")
    a = sub.add_parser("agree", help="exported model vs PyTorch FP32 (F1 at IoU 0.5)")
    a.add_argument("--backend", default="yoloe", choices=BACKENDS)
    a.add_argument("--weights", required=True)
    a.add_argument("--per-video", type=int, default=None)
    a.add_argument("--out", default=None)
    c = sub.add_parser("check-colour", help="which channel order reaches each network")
    c.add_argument("--out", default="outputs/virtual_cctv/ovdetect_colour_check.json")
    sub.add_parser("calib", help="write the INT8 calibration frames only")
    args = parser.parse_args(argv)
    # every weight file is already on disk: no Hugging Face Hub request (the stage-1 CLIP of openvocab otherwise
    # asks the Hub whether its cached LAION weights changed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    settings = load_settings()
    if settings.get("below_normal_priority", True):
        print(f"process priority: {lower_own_priority()}")

    if args.cmd == "ground":
        import openvocab
        index = openvocab.load_index(Path(args.db).parent)
        if index is None:
            print("no open-vocabulary index: run 'python openvocab.py build --db ...' first")
            return 1
        cams = _cameras_of(args.db)
        started = time.perf_counter()
        load_model(args.backend, settings, args.weights)
        print(f"{args.backend}: model loaded in {time.perf_counter() - started:.1f} s")
        started = time.perf_counter()                     # CLIP text encoder of stage 1 (loaded once per process)
        openvocab.embed_texts(["warm-up"], openvocab.encoder_settings(index, openvocab.load_settings()))
        print(f"stage-1 CLIP text encoder loaded in {time.perf_counter() - started:.1f} s")
        for text in args.text:
            stats: dict = {}
            started = time.perf_counter()
            hits = ground(index, text, cams, settings, backend=args.backend, weights=args.weights, stats=stats)
            total = (time.perf_counter() - started) * 1000
            print(f"Q: {text}  ({total:.0f} ms: clip {stats.get('stage1_ms')} ms, read {stats.get('read_ms')} ms, "
                  f"classes {stats.get('classes_ms')} ms, detect {stats.get('detect_ms')} ms on "
                  f"{stats.get('candidates')} frames)")
            if not hits:
                print("  no confident match")
            for h in hits:
                print(f"  {h['camera_id']} t={h['t']:.1f}s det {h['score']:.3f} clip#{h['clip_rank']} "
                      f"track {h['track_id']} box {[round(x) for x in h['box']]}")
                if args.evidence:
                    print(f"    evidence: {openvocab.write_evidence(h, cams[h['camera_id']], Path(args.db).parent, text)}")
        return 0
    if args.cmd == "vocab":
        started = time.perf_counter()
        build_vocab_index(args.db, args.backend, settings, weights=args.weights, force=args.force)
        print(f"done in {time.perf_counter() - started:.1f} s")
        return 0
    if args.cmd == "vocab-search":
        index = load_vocab_index(vocab_index_path(args.db, args.backend))
        if index is None:
            print("no vocabulary index: run 'python ovdetect.py vocab --db ...' first")
            return 1
        for text in args.text:
            term = vocab_term(text, settings)
            print(f"Q: {text}  -> term {term!r}")
            for h in (vocab_search(index, term, settings=settings) if term else []):
                print(f"  {h['camera_id']} t={h['t']:.1f}s conf {h['score']:.3f} track {h['track_id']}")
        return 0
    if args.cmd == "export":
        try:
            export_vocab_model(args.backend, args.format, args.int8, settings, out_dir=args.out_dir,
                               calib_max=args.calib_max)
        except Exception as exc:
            print(f"export failed for {args.backend}: {type(exc).__name__}: {exc}")
            return 1
        return 0
    if args.cmd == "bench":
        result = bench(args.backend, settings, args.weights, list(settings["vocabulary"]) if args.vocab else None,
                       args.frames)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "agree":
        result = agreement(args.backend, args.weights, settings, args.per_video)
        print(json.dumps(result))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "check-colour":
        results = [colour_check(backend, settings) for backend in BACKENDS]
        for r in results:
            print(json.dumps(r))
        Path(args.out).write_text(json.dumps({"checked_at": datetime.now().isoformat(timespec="seconds"),
                                              "results": results}, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "calib":
        print(prepare_calibration(settings))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
