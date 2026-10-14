"""Exported and INT8 person detectors for AquaWatch, measured against PyTorch FP32 (pre-registered rule R3).

    python quantize.py calib                                         # calibration frames + data yaml (data/calib/)
    python quantize.py export --weights yolo11n.pt --fmt onnx --int8 --imgsz 640
    python quantize.py check-threads --model models/quant/yolo11n_onnx_int8/yolo11n_int8.onnx --imgsz 640
    python quantize.py run-capped --video clip.mp4 --scenario pool --no-video    # run.py with the thread cap

What this module gives the rest of the project:
  * agreement(): how well a converted model reproduces the PyTorch FP32 boxes. Per frame, boxes are matched one to one
    (same class, highest IoU first, IoU >= 0.5); precision / recall / F1 of the test boxes against the reference boxes
    are summed over all frames. NumPy only.
  * r3_decision(): rule R3 of data/virtual_cctv/PREREGISTRATION_v3.md in code (INT8 only with F1 >= 0.98, else the
    FP32 export if it is faster, else PyTorch).
  * build_calibration(): about 200 frames spread evenly over videos that are NOT evaluation videos, written as an
    Ultralytics data yaml (images only; the INT8 calibration needs no labels).
  * export(): ONNX / OpenVINO, FP32 / INT8, with Ultralytics' own exporter (ONNX INT8 = ONNX Runtime static
    quantization of the Conv / MatMul layers; OpenVINO INT8 = NNCF post-training quantization with the detection head
    kept in float). Each artifact goes to models/quant/<stem>_<fmt>_<precision>/ with a manifest.json. The source .pt
    is copied to a work folder first, so it is never touched.
  * cap_runtime_threads(): Ultralytics creates ONNX Runtime sessions with no thread limit, compiles OpenVINO models
    with only PERFORMANCE_HINT=LATENCY (device AUTO when more than the CPU is visible), and its select_device() resets
    torch to min(8, cores - 1) = 7 threads on the first predict / export, overriding torch.set_num_threads(2).
    This caps all of them (and OpenCV) to N threads in this process; verify_thread_cap() measures it (process CPU
    time / wall time, plus the thread settings read back from the runtime). Around an exported model torch only does
    Ultralytics' pre / post-processing and gets 1 thread: with 2, its idle OpenMP worker kept spinning beside
    ONNX Runtime and the process used more than 2 cores.

How the pipeline gets the same cap: call cap_runtime_threads() once before the first YOLO(...) (tracker.py would need
two lines for that), or start run.py through `python quantize.py run-capped <run.py arguments>`, which installs the
cap in-process (torch at 1 thread when the configured model.weights is an export) and then calls run.main()
unchanged.

torch / ultralytics / cv2 / onnxruntime / openvino are imported inside functions only, so importing this is cheap.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __name__ == "__main__":                # CLI only: NumPy's OpenBLAS pool reads these once, when NumPy is loaded
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_var, "2")  # = QUANT_DEFAULTS["threads"]; torch / ORT / OpenVINO are capped at run time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # the project folder

# Defaults. An optional `quantize:` section of config.yaml overrides them (load_settings), like openvocab.py.
QUANT_DEFAULTS: dict[str, Any] = {
    "threads": 2,                         # CPU threads of every runtime (keeps the laptop usable)
    # torch threads for Ultralytics' pre / post-processing around an EXPORTED model (its runtime has the cap above).
    # With 2, torch's OpenMP worker keeps spinning (Intel OpenMP block time, 200 ms) while ONNX Runtime runs, so the
    # process used 2.1-2.6 cores instead of 2 (measured, see verify_thread_cap). 1 = no OpenMP worker at all.
    "export_torch_threads": 1,
    "below_normal_priority": True,        # the CLIs lower their own process priority
    "ov_device": "CPU",                   # OpenVINO device. Ultralytics would pick AUTO, which here includes the iGPU
    "ort_spinning": False,                # ONNX Runtime worker threads busy-wait between ops when True (burns CPU)
    "dynamic": True,                      # exports take any input shape, so they letterbox exactly like PyTorch
    "out_dir": "models/quant",            # <out_dir>/<stem>_<fmt>_<precision>/
    "work_dir": "outputs/_quant_check/work",   # temporary copy of the .pt while exporting
    # calibration: videos that are NOT evaluation videos. The 4 virtual cameras are crops of the excluded clip;
    # synthetic_pool.mp4 is a rendered cartoon (flat colours), not camera footage, so it is left out too.
    "calib_dir": "data/calib",
    "calib_videos": ["samples/uploads/*.mp4"],
    "calib_exclude": ["WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4", "synthetic_pool.mp4"],
    "calib_frames": 200,                  # spread evenly: the same number from every video
    "calib_resize_width": 640,            # saved at the processing width, like tracker.py (0 = original size)
    "calib_jpeg_quality": 95,
    # evaluation (R3): 20 evenly spaced frames from each of these videos, resized like tracker.py
    "eval_videos": ["data/virtual_cctv/videos/cam01.mp4", "data/virtual_cctv/videos/cam02.mp4",
                    "data/virtual_cctv/videos/cam03.mp4", "data/virtual_cctv/videos/cam04.mp4",
                    "samples/uploads/WhatsApp_Video_2026-10-07_at_2.36.05_PM.mp4"],
    "eval_frames_per_video": 20,
    "eval_resize_width": 640,             # config.yaml video.resize_width
    "conf": 0.25,                         # detection confidence threshold of every variant (R3)
    "nms_iou": 0.5,                       # NMS IoU of every variant (config.yaml model.iou)
    "match_iou": 0.5,                     # agreement: one-to-one match at IoU >= this (R3)
    "warmup": 5,                          # warm-up predictions before timing (not counted)
    "int8_min_f1": 0.98,                  # R3: INT8 adopted only with F1 >= this against PyTorch FP32
    "models": {                           # name -> weights, inference size, classes kept, class-agnostic NMS
        "yolo11n": {"weights": "yolo11n.pt", "imgsz": 640, "classes": [0], "agnostic_nms": False},
        "yolo11s": {"weights": "yolo11s.pt", "imgsz": 960, "classes": [0], "agnostic_nms": False},
        "best": {"weights": "training/runs/swimmer/weights/best.pt", "imgsz": 640, "classes": [0, 1, 2, 3, 4],
                 "agnostic_nms": True},   # state classes of one swimmer: tracker.py uses class-agnostic NMS
    },
    "eval_out": "outputs/quant/quant_eval.json",
}
FORMATS = ("onnx", "openvino")
PRECISIONS = ("fp32", "int8")
VARIANTS = ("pt_fp32", "onnx_fp32", "onnx_int8", "openvino_fp32", "openvino_int8")
REFERENCE = "pt_fp32"
MANIFEST = "manifest.json"
CALIB_SPLIT = "calib"                     # data/calib/images/calib/*.jpg
CALIB_YAML = "calib.yaml"
CALIB_MANIFEST = "calib_manifest.json"
CAP_TOLERANCE = 1.10                      # cap "holds" when CPU time / wall time <= threads x this (stray threads)

_CAP_STATE: dict[str, Any] = {}           # originals of the patched runtime entry points + the current cap


def load_settings(overrides: Mapping[str, Any] | None = None, use_config: bool = True) -> dict:
    """QUANT_DEFAULTS, then config.yaml's optional `quantize:` section, then the non-None overrides."""
    settings = dict(QUANT_DEFAULTS)
    if use_config:
        try:
            import utils
            settings.update(utils.load_config().get("quantize") or {})
        except Exception:                                     # a broken config must not stop the tool
            pass
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return settings


def resolve(path: str | Path) -> Path:
    """Absolute path; relative paths are taken from the project folder."""
    p = Path(str(path))
    return p if p.is_absolute() else ROOT / p


def rel(path: str | Path) -> str:
    """Path relative to the project folder (posix) when it is inside it, else the absolute path."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------- agreement (NumPy only)

def _as_boxes(frame: Iterable | None) -> np.ndarray:
    """One frame's boxes as an N x 6 float array (cls, conf, x1, y1, x2, y2)."""
    arr = np.asarray(list(frame) if frame is not None else [], dtype=float)
    return arr.reshape(-1, 6) if arr.size else np.zeros((0, 6))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box of `a` (N x 4, x1 y1 x2 y2) with every box of `b` (M x 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def match_frame(ref: np.ndarray, test: np.ndarray, iou_thr: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matches (ref index, test index, IoU) in one frame: same class only, highest IoU first,
    IoU >= iou_thr. Ties keep the lower indices first (stable sort), so the result is deterministic."""
    if not len(ref) or not len(test):
        return []
    iou = iou_matrix(ref[:, 2:6], test[:, 2:6])
    same_class = np.rint(ref[:, 0])[:, None] == np.rint(test[:, 0])[None, :]
    iou = np.where(same_class, iou, -1.0)
    pairs = np.argwhere(iou >= iou_thr)
    order = np.argsort(-iou[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_ref, used_test, matches = set(), set(), []
    for i, j in pairs[order]:
        if i in used_ref or j in used_test:
            continue
        used_ref.add(int(i))
        used_test.add(int(j))
        matches.append((int(i), int(j), float(iou[i, j])))
    return matches


def agreement(ref_frames: Sequence, test_frames: Sequence, iou_thr: float = 0.5) -> dict:
    """How well the test boxes reproduce the reference boxes over all frames.

    Inputs: one list of boxes (cls, conf, x1, y1, x2, y2) per frame, the same frames in the same order. Returns
    {"precision", "recall", "f1", "n_ref", "n_test", "matched", "mean_iou"}: matched / n_test, matched / n_ref and
    2 * matched / (n_ref + n_test) (the harmonic mean of the two). Nothing on either side counts as perfect agreement
    (1.0); precision is 1.0 when there are no test boxes, recall 1.0 when there are no reference boxes."""
    if len(ref_frames) != len(test_frames):
        raise ValueError(f"agreement needs the same frames: {len(ref_frames)} reference vs {len(test_frames)} test")
    n_ref = n_test = matched = 0
    ious: list[float] = []
    for ref, test in zip(ref_frames, test_frames):
        r, t = _as_boxes(ref), _as_boxes(test)
        n_ref += len(r)
        n_test += len(t)
        found = match_frame(r, t, iou_thr)
        matched += len(found)
        ious.extend(m[2] for m in found)
    return {"precision": matched / n_test if n_test else 1.0,
            "recall": matched / n_ref if n_ref else 1.0,
            "f1": 2.0 * matched / (n_ref + n_test) if (n_ref + n_test) else 1.0,
            "n_ref": n_ref, "n_test": n_test, "matched": matched,
            "mean_iou": float(np.mean(ious)) if ious else None}


# ---------------------------------------------------------------------------- rule R3 (pure Python)

def _usable(row: Mapping[str, Any] | None) -> bool:
    """A variant with a measured F1 and median latency and no error."""
    return bool(row) and not row.get("error") and row.get("f1") is not None and row.get("ms_median") is not None


FP32_TIE = 1.10            # Addendum A of the pre-registration: FP32 exports within 10 % of each other are a tie


def r3_decision(results: Mapping[str, Mapping[str, Any] | None], min_f1: float = 0.98) -> dict:
    """Pre-registered rule R3 for one model, as written in PREREGISTRATION_v3.md.

    `results` maps a variant name (pt_fp32, onnx_fp32, onnx_int8, openvino_fp32, openvino_int8) to
    {"f1", "ms_median"} (missing / None / "error" = not measured). The rule, in order:
      1. "INT8 is adopted only if F1 >= min_f1" against PyTorch FP32. Addendum A of the pre-registration (written
         before the final run) reads this as a NECESSARY condition: an INT8 variant must also be faster than PyTorch,
         else it brings nothing. Of the INT8 variants that pass both, the faster is adopted;
      2. "otherwise the FP32 export (numerically the same model) is used if it is faster" than PyTorch: the faster
         of ONNX / OpenVINO FP32. An FP32 export below F1 min_f1 is not "numerically the same model" (a broken
         export), so it is not used;
      3. otherwise PyTorch stays.
    Returns {"choice", "reason", "fastest", "note"}."""
    ref = results.get(REFERENCE) or {}
    ref_ms = ref.get("ms_median") if not ref.get("error") else None
    measured = {name: row for name, row in results.items() if _usable(row)}
    fastest = min(measured, key=lambda n: measured[n]["ms_median"]) if measured else None
    if ref_ms is None:
        return {"choice": REFERENCE, "reason": "PyTorch FP32 reference not measured: nothing to compare, PyTorch stays",
                "fastest": fastest, "note": None}

    def described(name: str) -> str:
        row = results.get(name) or {}
        if not _usable(row):
            return f"{name} not measured" + (f" ({row.get('error')})" if row.get("error") else "")
        return f"{name} F1 {row['f1']:.4f}, {row['ms_median']:.1f} ms"

    int8_ok = [n for n in ("onnx_int8", "openvino_int8") if n in measured and measured[n]["f1"] >= min_f1
               and measured[n]["ms_median"] < ref_ms]
    if int8_ok:
        choice = min(int8_ok, key=lambda n: measured[n]["ms_median"])
        choice_ms = measured[choice]["ms_median"]
        notes = []
        if choice_ms >= ref_ms:
            notes.append(f"the adopted INT8 is not faster than PyTorch ({choice_ms:.1f} vs {ref_ms:.1f} ms)")
        faster_fp32 = [n for n in ("onnx_fp32", "openvino_fp32") if n in measured and measured[n]["f1"] >= min_f1
                       and measured[n]["ms_median"] < choice_ms]
        if faster_fp32:
            notes.append(f"{', '.join(faster_fp32)} is faster than the adopted INT8 (R3 puts INT8 first)")
        return {"choice": choice, "reason": f"{described(choice)}: F1 >= {min_f1} against PyTorch FP32 "
                f"({ref_ms:.1f} ms)", "fastest": fastest, "note": "; ".join(notes) or None}
    why_int8 = "; ".join(described(n) for n in ("onnx_int8", "openvino_int8"))
    fp32_ok = [n for n in ("onnx_fp32", "openvino_fp32")
               if n in measured and measured[n]["f1"] >= min_f1 and measured[n]["ms_median"] < ref_ms]
    if fp32_ok:
        choice = min(fp32_ok, key=lambda n: measured[n]["ms_median"])
        if "openvino_fp32" in fp32_ok and choice != "openvino_fp32" and                 measured["openvino_fp32"]["ms_median"] <= measured[choice]["ms_median"] * FP32_TIE:
            choice = "openvino_fp32"          # Addendum A: within 10 % = a tie, OpenVINO (thread cap verified) wins
        return {"choice": choice, "reason": f"no INT8 variant reached F1 {min_f1} ({why_int8}); {described(choice)} "
                f"is faster than PyTorch ({ref_ms:.1f} ms)", "fastest": fastest, "note": None}
    why_fp32 = "; ".join(described(n) for n in ("onnx_fp32", "openvino_fp32"))
    return {"choice": REFERENCE, "reason": f"no INT8 variant reached F1 {min_f1} ({why_int8}) and no FP32 export is "
            f"faster than PyTorch at F1 >= {min_f1} ({why_fp32}); PyTorch ({ref_ms:.1f} ms) stays",
            "fastest": fastest, "note": None}


# ---------------------------------------------------------------------------- frames (cv2 inside functions)

def sample_indices(n_frames: int, n: int) -> list[int]:
    """`n` frame indices spread evenly over a video of `n_frames` frames (the middle of n equal parts)."""
    n = max(0, min(int(n), int(n_frames)))
    return [int((i + 0.5) * n_frames / n) for i in range(n)]


def count_frames(video: str | Path) -> int:
    """Exact number of decodable frames (the container's frame count can be wrong)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_frames(video: str | Path, indices: Iterable[int], resize_width: int = 0) -> list[tuple[int, np.ndarray]]:
    """(index, BGR frame) for the wanted indices, read sequentially (exact on any codec) and resized to
    `resize_width` pixels wide exactly like tracker.py does before detection (0 = original size)."""
    import cv2
    import tracker
    wanted = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {video}")
    out, index, pos = [], 0, 0
    try:
        while pos < len(wanted) and cap.grab():
            if index == wanted[pos]:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    if resize_width:
                        frame = tracker.resize_frame(frame, tracker.proc_size_for(
                            (frame.shape[1], frame.shape[0]), resize_width))
                    out.append((index, frame))
                pos += 1
            index += 1
    finally:
        cap.release()
    return out


def evaluation_frames(settings: Mapping[str, Any] | None = None, per_video: int | None = None) -> list[dict]:
    """The R3 evaluation frames: `per_video` evenly spaced frames of every eval video (resized like tracker.py).
    [{"video": relative path, "frame": index, "image": BGR array}]."""
    s = load_settings(settings)
    per_video = int(per_video or s["eval_frames_per_video"])
    frames = []
    for video in s["eval_videos"]:
        path = resolve(video)
        for index, image in read_frames(path, sample_indices(count_frames(path), per_video),
                                        int(s["eval_resize_width"] or 0)):
            frames.append({"video": rel(path), "frame": index, "image": image})
    return frames


# ---------------------------------------------------------------------------- calibration frames

def _file_stamp(path: Path) -> str:
    """Size + mtime of a file: cheap change detection for the caches."""
    stat = Path(path).stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _sha1(path: str | Path) -> str:
    """SHA-1 of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def calibration_videos(settings: Mapping[str, Any] | None = None) -> list[Path]:
    """Videos for INT8 calibration: the calib_videos globs minus calib_exclude (file names), and minus every video
    with the same content as an evaluation video or as a video already taken (copies in two folders)."""
    s = load_settings(settings)
    excluded = {name.lower() for name in s["calib_exclude"]}
    eval_hashes = {_sha1(resolve(v)) for v in s["eval_videos"] if resolve(v).is_file()}
    picked, seen = [], set(eval_hashes)
    for pattern in s["calib_videos"]:
        for path in sorted(glob.glob(str(resolve(pattern)))):
            path = Path(path)
            if path.name.lower() in excluded or not path.is_file():
                continue
            digest = _sha1(path)
            if digest in seen:
                continue
            seen.add(digest)
            picked.append(path)
    return picked


def build_calibration(settings: Mapping[str, Any] | None = None, force: bool = False, echo=print) -> dict:
    """Write about `calib_frames` frames (the same number from every calibration video, evenly spaced) to
    <calib_dir>/images/calib/ and an Ultralytics data yaml next to them. Cached: rebuilt only when the videos or the
    settings change. Returns {"yaml", "n_images", "videos", "id"}."""
    s = load_settings(settings)
    calib_dir = resolve(s["calib_dir"])
    img_dir = calib_dir / "images" / CALIB_SPLIT
    yaml_path, manifest_path = calib_dir / CALIB_YAML, calib_dir / CALIB_MANIFEST
    videos = calibration_videos(s)
    if not videos:
        raise RuntimeError("no calibration videos found (check quantize.calib_videos / calib_exclude)")
    keyed = {k: s[k] for k in ("calib_videos", "calib_exclude", "calib_frames", "calib_resize_width",
                               "calib_jpeg_quality")}
    stamps = {rel(v): _file_stamp(v) for v in videos}
    if not force and manifest_path.is_file() and yaml_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (old.get("settings") == keyed and old.get("stamps") == stamps
                    and all((calib_dir / f).is_file() for f in old.get("files", []))):
                echo(f"calibration set unchanged: {old['n_images']} images in {rel(img_dir)}")
                return {"yaml": str(yaml_path), "n_images": old["n_images"], "videos": old["videos"], "id": old["id"]}
        except (OSError, ValueError, KeyError):
            pass
    import cv2
    shutil.rmtree(calib_dir / "images", ignore_errors=True)          # only ever our own generated folders
    shutil.rmtree(calib_dir / "labels", ignore_errors=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    (calib_dir / "labels").mkdir(parents=True, exist_ok=True)         # lets Ultralytics write its label cache
    total = int(s["calib_frames"])
    per = [total // len(videos) + (1 if i < total % len(videos) else 0) for i in range(len(videos))]
    files, used = [], []
    for video, n in zip(videos, per):
        frames = read_frames(video, sample_indices(count_frames(video), n), int(s["calib_resize_width"] or 0))
        for index, image in frames:
            name = f"images/{CALIB_SPLIT}/{video.stem}_{index:06d}.jpg"
            cv2.imwrite(str(calib_dir / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(s["calib_jpeg_quality"])])
            files.append(name)
        used.append({"video": rel(video), "frames": [i for i, _ in frames]})
        echo(f"  {rel(video)}: {len(frames)} calibration frames")
    yaml_path.write_text(
        "# INT8 calibration images for quantize.py (written by `python quantize.py calib`).\n"
        "# Frames of videos that are NOT evaluation videos; no labels: calibration only needs the images.\n"
        "# 'path' is left out on purpose: Ultralytics then resolves the folders next to this file.\n"
        f"train: images/{CALIB_SPLIT}\nval: images/{CALIB_SPLIT}\nnames:\n  0: object\n", encoding="utf-8")
    digest = hashlib.sha1("".join(sorted(files)).encode() + json.dumps(stamps, sort_keys=True).encode()).hexdigest()
    manifest = {"id": digest[:12], "settings": keyed, "stamps": stamps, "videos": used, "n_images": len(files),
                "files": files, "created": datetime.now().isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    echo(f"calibration set: {len(files)} images from {len(videos)} videos -> {rel(yaml_path)}")
    return {"yaml": str(yaml_path), "n_images": len(files), "videos": used, "id": manifest["id"]}


# ---------------------------------------------------------------------------- thread cap

def _patch_onnxruntime(ort) -> None:
    """Replace onnxruntime.InferenceSession with a subclass that caps intra-op threads (and turns spinning off).
    Ultralytics' backend and ONNX Runtime's quantization calibrator both look the class up at call time."""
    if "ort_session" not in _CAP_STATE:
        _CAP_STATE["ort_session"] = ort.InferenceSession
    base = _CAP_STATE["ort_session"]
    if getattr(ort.InferenceSession, "_aquawatch_capped", False):
        return

    class CappedInferenceSession(base):
        """onnxruntime.InferenceSession that never uses more than the capped number of threads."""
        _aquawatch_capped = True

        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
            options = sess_options if sess_options is not None else ort.SessionOptions()
            cap = int(_CAP_STATE.get("threads") or 2)
            if options.intra_op_num_threads <= 0 or options.intra_op_num_threads > cap:
                options.intra_op_num_threads = cap
            if options.inter_op_num_threads <= 0 or options.inter_op_num_threads > cap:
                options.inter_op_num_threads = 1
            if not _CAP_STATE.get("ort_spinning"):
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            super().__init__(path_or_bytes, options, providers, provider_options, **kwargs)

    ort.InferenceSession = CappedInferenceSession


def _patch_openvino(ov) -> None:
    """Wrap openvino.Core.compile_model: AUTO / no device -> the configured device (CPU), and on CPU
    INFERENCE_NUM_THREADS = the cap. Ultralytics' backend and NNCF's calibration both compile through it."""
    if "ov_compile" not in _CAP_STATE:
        _CAP_STATE["ov_compile"] = ov.Core.compile_model
    if getattr(ov.Core.compile_model, "_aquawatch_capped", False):
        return
    original = _CAP_STATE["ov_compile"]

    def compile_model(core, model, device_name=None, config=None, **kwargs):
        """openvino.Core.compile_model with the AquaWatch thread cap."""
        device = device_name
        forced = _CAP_STATE.get("ov_device")
        if forced and (device is None or str(device).upper() == "AUTO"):
            device = forced
        cfg = dict(config or {})
        if device is not None and str(device).upper().startswith("CPU"):
            cap = int(_CAP_STATE.get("threads") or 2)
            current = int(cfg.get("INFERENCE_NUM_THREADS") or 0)
            if current <= 0 or current > cap:
                cfg["INFERENCE_NUM_THREADS"] = cap
        if device is None:
            return original(core, model, config=cfg, **kwargs)
        return original(core, model, device, cfg, **kwargs)

    compile_model._aquawatch_capped = True
    ov.Core.compile_model = compile_model


def _patch_ultralytics(n: int) -> int | None:
    """Ultralytics' select_device() calls torch.set_num_threads(NUM_THREADS) (= min(8, cores - 1)) on every predictor /
    exporter set-up, which silently undoes torch.set_num_threads(2). Point that module constant at the cap."""
    _ultralytics_env()
    try:
        from ultralytics.utils import torch_utils
    except ImportError:
        return None
    if "ul_num_threads" not in _CAP_STATE:
        _CAP_STATE["ul_num_threads"] = torch_utils.NUM_THREADS
    torch_utils.NUM_THREADS = n
    return torch_utils.NUM_THREADS


def is_exported(path: str | Path) -> bool:
    """True for an exported model (.onnx file, *_openvino_model folder, .xml), False for a PyTorch .pt."""
    p = Path(str(path))
    return p.suffix.lower() in (".onnx", ".xml") or p.name.endswith("_openvino_model") or p.is_dir()


def cap_runtime_threads(threads: int | None = None, settings: Mapping[str, Any] | None = None,
                        torch_threads: int | None = None) -> dict:
    """Limit ONNX Runtime, OpenVINO, torch (also after Ultralytics' select_device) and OpenCV in THIS process to
    `threads` threads. `torch_threads` (default = threads) is the torch share: use 1 around an exported model, whose
    runtime does the heavy work (see "export_torch_threads"). Call it before a model is loaded (sessions that
    already exist keep their threads). Idempotent. Returns what was capped."""
    s = load_settings(settings)
    n = max(1, int(threads or s["threads"]))
    nt = max(1, min(n, int(torch_threads or n)))
    _CAP_STATE.update({"threads": n, "ov_device": s.get("ov_device") or None,
                       "ort_spinning": bool(s.get("ort_spinning"))})
    info: dict[str, Any] = {"threads": n}
    try:
        import torch
        torch.set_num_threads(nt)
        try:
            torch.set_num_interop_threads(1)                    # only allowed before the first parallel work
        except RuntimeError:
            pass
        info["torch"] = torch.get_num_threads()
    except ImportError:
        info["torch"] = None
    info["ultralytics_select_device_threads"] = _patch_ultralytics(nt)  # importing ultralytics sets OpenCV to 0
    try:
        import cv2
        if cv2.getNumThreads() > n:                             # 1 = sequential (Ultralytics' setNumThreads(0))
            cv2.setNumThreads(n)
        info["opencv"] = cv2.getNumThreads()
    except ImportError:
        info["opencv"] = None
    try:
        import onnxruntime as ort
        _patch_onnxruntime(ort)
        info["onnxruntime"] = f"intra_op_num_threads={n}, spinning={'on' if _CAP_STATE['ort_spinning'] else 'off'}"
    except ImportError:
        info["onnxruntime"] = None
    try:
        import openvino as ov
        _patch_openvino(ov)
        info["openvino"] = f"INFERENCE_NUM_THREADS={n} on {_CAP_STATE['ov_device'] or 'the device Ultralytics picks'}"
    except ImportError:
        info["openvino"] = None
    return info


def uncap_runtime_threads() -> None:
    """Undo the ONNX Runtime / OpenVINO patches (tests, or a baseline measurement)."""
    try:
        import onnxruntime as ort
        if "ort_session" in _CAP_STATE:
            ort.InferenceSession = _CAP_STATE.pop("ort_session")
    except ImportError:
        pass
    try:
        import openvino as ov
        if "ov_compile" in _CAP_STATE:
            ov.Core.compile_model = _CAP_STATE.pop("ov_compile")
    except ImportError:
        pass
    if "ul_num_threads" in _CAP_STATE:
        try:
            from ultralytics.utils import torch_utils
            torch_utils.NUM_THREADS = _CAP_STATE.pop("ul_num_threads")
        except ImportError:
            pass
    _CAP_STATE.pop("threads", None)


def thread_readback(model) -> dict:
    """The thread settings the loaded runtime actually uses (after the first predict created it)."""
    predictor = getattr(model, "predictor", None)
    backend = getattr(getattr(predictor, "model", None), "backend", None)
    try:
        import torch
        torch_threads = torch.get_num_threads()               # pre / post-processing share of an exported model
        if backend is not None and hasattr(backend, "session"):
            opts = backend.session.get_session_options()
            try:
                spinning = opts.get_session_config_entry("session.intra_op.allow_spinning")
            except Exception:                                  # not set = ONNX Runtime's default (spinning on)
                spinning = "default (1)"
            return {"runtime": "onnxruntime", "intra_op_num_threads": int(opts.intra_op_num_threads),
                    "inter_op_num_threads": int(opts.inter_op_num_threads), "allow_spinning": spinning,
                    "providers": list(backend.session.get_providers()), "torch_threads": torch_threads}
        if backend is not None and hasattr(backend, "ov_compiled_model"):
            compiled = backend.ov_compiled_model
            out = {"runtime": "openvino", "EXECUTION_DEVICES": list(compiled.get_property("EXECUTION_DEVICES"))}
            for key in ("INFERENCE_NUM_THREADS", "NUM_STREAMS", "PERFORMANCE_HINT"):
                try:
                    out[key] = str(compiled.get_property(key))
                except Exception:                              # AUTO does not expose every CPU property
                    out[key] = None
            out["torch_threads"] = torch_threads
            return out
        return {"runtime": "torch", "num_threads": torch_threads,
                "num_interop_threads": torch.get_num_interop_threads()}
    except Exception as exc:                                  # a read-back failure must not stop a measurement
        return {"runtime": "unknown", "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------- models (ultralytics inside functions)

def _ultralytics_env() -> None:
    """Environment for Ultralytics: the project's settings folder, no automatic pip installs."""
    shared = ROOT.parent / ".ultralytics"
    if shared.is_dir():
        os.environ.setdefault("YOLO_CONFIG_DIR", str(shared))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def _quiet_settings() -> None:
    """The exporter flips settings['openvino_msg'] once and SAVES settings.json; set it in memory only."""
    try:
        from ultralytics.utils import SETTINGS
        dict.__setitem__(SETTINGS, "openvino_msg", False)
    except Exception:
        pass


def load_detector(path: str | Path, threads: int | None = None, task: str = "detect",
                  settings: Mapping[str, Any] | None = None):
    """A YOLO model (.pt, .onnx file or *_openvino_model folder) whose runtime uses at most `threads` threads.
    The cap is process-wide (see cap_runtime_threads); this is the one-call way to load an exported detector.
    For an exported model torch (pre / post-processing only) gets export_torch_threads threads."""
    s = load_settings(settings)
    cap_runtime_threads(threads or s["threads"], s,
                        torch_threads=int(s["export_torch_threads"]) if is_exported(resolve(path)) else None)
    _ultralytics_env()
    from ultralytics import YOLO
    return YOLO(str(resolve(path)), task=task)


def result_boxes(result, min_conf: float = 0.0) -> list[tuple]:
    """(cls, conf, x1, y1, x2, y2) of every box of one Ultralytics Results object with conf >= min_conf."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    cls = boxes.cls.cpu().numpy().reshape(-1)
    conf = boxes.conf.cpu().numpy().reshape(-1)
    xyxy = boxes.xyxy.cpu().numpy().reshape(-1, 4)
    return [(int(c), round(float(p), 4), *(round(float(v), 2) for v in box))
            for c, p, box in zip(cls, conf, xyxy) if p >= min_conf]


def size_mb(path: str | Path) -> float | None:
    """Size of a file, or of every file in a folder, in MB (2^20 bytes); None when missing."""
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / 2 ** 20, 2)
    if p.is_dir():
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2 ** 20, 2)
    return None


def peak_memory_mb() -> float | None:
    """Peak memory of this process so far in MB (Windows peak working set, else max RSS); None when unknown."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak is None:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return round(peak / 2 ** 20, 1)
    except Exception:                                         # optional number: never fail for it
        return None


def predict_kwargs(model_cfg: Mapping[str, Any], settings: Mapping[str, Any] | None = None) -> dict:
    """Keyword arguments of model.predict for one model of settings["models"] (R3: conf 0.25, the NMS IoU of
    config.yaml, its classes and class-agnostic NMS like tracker.py, CPU, single image)."""
    s = load_settings(settings)
    return {"imgsz": int(model_cfg["imgsz"]), "conf": float(s["conf"]), "iou": float(s["nms_iou"]),
            "classes": [int(c) for c in model_cfg["classes"]], "agnostic_nms": bool(model_cfg.get("agnostic_nms")),
            "device": "cpu", "verbose": False}


def _versions() -> dict:
    """Installed versions of the libraries an export depends on (read from metadata, nothing imported)."""
    from importlib import metadata
    out = {}
    for name in ("ultralytics", "torch", "onnx", "onnxruntime", "onnxslim", "openvino", "nncf"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def artifact_dir(weights: str | Path, fmt: str, precision: str, settings: Mapping[str, Any] | None = None) -> Path:
    """models/quant/<stem>_<fmt>_<precision>/ of a source .pt."""
    s = load_settings(settings)
    return resolve(s["out_dir"]) / f"{Path(str(weights)).stem}_{fmt}_{precision}"


def read_manifest(weights: str | Path, fmt: str, precision: str,
                  settings: Mapping[str, Any] | None = None) -> dict | None:
    """The manifest.json of an export, or None."""
    path = artifact_dir(weights, fmt, precision, settings) / MANIFEST
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_artifact(weights: str | Path, fmt: str, precision: str,
                  settings: Mapping[str, Any] | None = None) -> Path | None:
    """The loadable model path of an existing export (.onnx file or *_openvino_model folder), or None."""
    manifest = read_manifest(weights, fmt, precision, settings)
    if not manifest or not manifest.get("model_path"):
        return None
    path = resolve(manifest["model_path"])
    return path if path.exists() else None


def export(weights: str | Path, fmt: str = "onnx", int8: bool = False, imgsz: int = 640,
           settings: Mapping[str, Any] | None = None, force: bool = False, echo=print) -> dict:
    """Export a YOLO .pt to ONNX or OpenVINO, FP32 or INT8, into models/quant/<stem>_<fmt>_<precision>/.

    Uses Ultralytics' exporter on a COPY of the .pt (in work_dir), so the source is never modified. INT8 calibrates
    on the calibration set (build_calibration). Reused when the manifest says the same source, size, shape mode and
    calibration set. Returns the manifest: {"model_path", "size_mb", "export_s", "export_cpu_per_wall", ...}."""
    s = load_settings(settings)
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {FORMATS}, not {fmt!r}")
    src = resolve(weights)
    if not src.is_file():
        raise FileNotFoundError(f"weights not found: {src}")
    precision = "int8" if int8 else "fp32"
    out = artifact_dir(src, fmt, precision, s)
    want: dict[str, Any] = {"source": rel(src), "source_sha1": _sha1(src), "fmt": fmt, "precision": precision,
                            "imgsz": int(imgsz), "dynamic": bool(s["dynamic"])}
    calib = build_calibration(s, echo=echo) if int8 else None
    if calib:
        want["calibration_id"] = calib["id"]
    old = read_manifest(src, fmt, precision, s)
    if (not force and old and all(old.get(k) == v for k, v in want.items())
            and old.get("model_path") and resolve(old["model_path"]).exists()):
        echo(f"{out.name}: up to date ({old['model_path']}), not rebuilt")
        return old

    cap_runtime_threads(s["threads"], s)                       # calibration runs under the same cap
    _ultralytics_env()
    from ultralytics import YOLO
    _quiet_settings()
    work = resolve(s["work_dir"]) / out.name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    copy = work / src.name
    shutil.copy2(src, copy)
    kwargs: dict[str, Any] = {"format": fmt, "imgsz": int(imgsz), "dynamic": bool(s["dynamic"]), "device": "cpu",
                              "batch": 1}
    if int8:
        kwargs.update(quantize=8, data=calib["yaml"])
    echo(f"{out.name}: exporting {rel(src)} ({', '.join(f'{k}={v}' for k, v in kwargs.items())})")
    # check_det_dataset() downloads Arial.ttf for plotting when it is missing; an export plots nothing, so skip it
    import ultralytics.data.utils as data_utils
    font_check = data_utils.check_font
    data_utils.check_font = lambda *args, **kwargs: None
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        produced = Path(str(YOLO(str(copy)).export(**kwargs)))
    finally:
        data_utils.check_font = font_check
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    if not produced.exists():
        raise RuntimeError(f"Ultralytics reported {produced} but it does not exist")
    shutil.rmtree(out, ignore_errors=True)                     # our own folder under models/quant
    out.mkdir(parents=True, exist_ok=True)
    target = out / produced.name
    shutil.move(str(produced), str(target))
    method = {("onnx", "fp32"): "Ultralytics ONNX export (FP32)",
              ("onnx", "int8"): "Ultralytics ONNX export + ONNX Runtime quantize_static (QDQ, MinMax, Conv/Gemm/MatMul "
                                "INT8, rest float)",
              ("openvino", "fp32"): "Ultralytics OpenVINO export (FP32 IR)",
              ("openvino", "int8"): "Ultralytics OpenVINO export + NNCF quantize (MIXED preset, detection head kept "
                                    "float)"}[(fmt, precision)]
    ratio = cpu / wall if wall > 0 else None
    manifest = {**want, "model_path": rel(target), "size_mb": size_mb(target), "method": method,
                "export_s": round(wall, 1), "export_cpu_s": round(cpu, 1),
                "export_cpu_per_wall": round(ratio, 2) if ratio is not None else None, "threads": int(s["threads"]),
                "cap_held": (ratio <= int(s["threads"]) * CAP_TOLERANCE) if ratio is not None else None,
                "process_peak_mb": peak_memory_mb(),
                "calibration": ({"yaml": rel(calib["yaml"]), "n_images": calib["n_images"],
                                 "videos": [v["video"] for v in calib["videos"]]} if calib else None),
                "versions": _versions(), "created": datetime.now().isoformat(timespec="seconds")}
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    echo(f"{out.name}: {manifest['model_path']} ({manifest['size_mb']} MB) in {wall:.1f} s, "
         f"process CPU / wall = {manifest['export_cpu_per_wall']}")
    return manifest


# ---------------------------------------------------------------------------- verifying the cap

def verify_thread_cap(model_path: str | Path, threads: int | None = None, imgsz: int = 640, n: int = 20,
                      capped: bool = True, settings: Mapping[str, Any] | None = None) -> dict:
    """Run `n` single-frame predictions (after 3 warm-ups) and compare the process CPU time with the wall time:
    with the cap holding, CPU / wall stays <= threads (x CAP_TOLERANCE for stray threads). `capped=False` measures
    the stock Ultralytics behaviour for comparison (torch is still capped, only ONNX Runtime / OpenVINO are not)."""
    s = load_settings(settings)
    n_threads = int(threads or s["threads"])
    if capped:
        model = load_detector(model_path, n_threads, settings=s)
    else:
        uncap_runtime_threads()
        import torch
        torch.set_num_threads(n_threads)
        _ultralytics_env()
        from ultralytics import YOLO
        model = YOLO(str(resolve(model_path)), task="detect")
    frame = read_frames(resolve(s["eval_videos"][0]), [0], int(s["eval_resize_width"] or 0))[0][1]
    kw = {"imgsz": int(imgsz), "conf": float(s["conf"]), "iou": float(s["nms_iou"]), "device": "cpu",
          "verbose": False}
    for _ in range(3):
        model.predict(frame, **kw)
    wall0, cpu0 = time.perf_counter(), time.process_time()
    for _ in range(int(n)):
        model.predict(frame, **kw)
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    ratio = cpu / wall if wall > 0 else float("nan")
    return {"model": rel(resolve(model_path)), "capped": capped, "threads": n_threads, "frames": int(n),
            "wall_s": round(wall, 3), "cpu_s": round(cpu, 3), "cpu_per_wall": round(ratio, 2),
            "ms_per_frame": round(1000 * wall / max(1, n), 1), "readback": thread_readback(model),
            "holds": (ratio <= n_threads * CAP_TOLERANCE) if capped else None}


def lower_priority() -> str:
    """Put this process at below-normal priority (Windows) / nice +5; returns what was done."""
    try:
        import psutil
        proc = psutil.Process()
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(max(proc.nice(), 5))
        return "below normal"
    except Exception as exc:                                  # optional: never stop the tool for this
        return f"unchanged ({type(exc).__name__})"


def set_torch_share(n: int) -> None:
    """torch threads = n now AND for every Ultralytics predictor created later (select_device resets them)."""
    import torch
    torch.set_num_threads(max(1, int(n)))
    _patch_ultralytics(max(1, int(n)))


def run_capped(argv: Sequence[str]) -> int:
    """run.py's main() in this process with the runtime thread cap installed first, so tracker.py's YOLO(...) of an
    ONNX / OpenVINO model is capped too. When model.weights is an export, torch gets export_torch_threads threads
    only while tracker.track_video runs (pre / post-processing); the PyTorch pose model of the pool stage keeps
    `threads`. Prints the process CPU time / wall time at the end."""
    s = load_settings()
    sys.path.insert(0, str(ROOT))
    import run
    import tracker
    torch_threads = None
    try:                                                      # which detector will tracker.py load?
        args, _ = run.build_parser().parse_known_args(list(argv))
        import utils
        weights = utils.load_config(args.config, args.scenario)["model"]["weights"]
        if is_exported(resolve(weights)):
            torch_threads = int(s["export_torch_threads"])
    except Exception:                                         # run.main() reports bad arguments itself
        pass
    print(f"thread cap: {cap_runtime_threads(s['threads'], s)}"
          + (f"; torch {torch_threads} thread(s) while tracking with the exported detector" if torch_threads else ""),
          flush=True)
    original_track_video = tracker.track_video

    def track_video(*a, **kw):
        """tracker.track_video with torch at export_torch_threads for an export (run.py imports it at call time),
        plus the CPU time / wall time of the tracking stage alone."""
        if torch_threads:
            set_torch_share(torch_threads)
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            return original_track_video(*a, **kw)
        finally:
            w, c = time.perf_counter() - t0, time.process_time() - c0
            print(f"run-capped: tracking stage CPU {c:.1f} s / wall {w:.1f} s = {c / max(w, 1e-9):.2f} cores busy "
                  f"(cap {s['threads']})", flush=True)
            if torch_threads:
                set_torch_share(s["threads"])
    tracker.track_video = track_video
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        code = run.main(list(argv))
    finally:
        tracker.track_video = original_track_video
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    print(f"run-capped: process CPU {cpu:.1f} s / wall {wall:.1f} s = {cpu / max(wall, 1e-9):.2f} cores busy "
          f"on average (cap {s['threads']})", flush=True)
    return int(code or 0)


# ---------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    """calib / export / check-threads / run-capped from the command line."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run-capped":                      # everything after it goes to run.py unchanged
        if load_settings()["below_normal_priority"]:
            lower_priority()
        return run_capped(argv[1:])
    parser = argparse.ArgumentParser(description="AquaWatch exported / INT8 detectors (rule R3)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calib", help="extract the INT8 calibration frames")
    c.add_argument("--force", action="store_true")
    e = sub.add_parser("export", help="export one model")
    e.add_argument("--weights", required=True)
    e.add_argument("--fmt", choices=FORMATS, required=True)
    e.add_argument("--int8", action="store_true")
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--force", action="store_true")
    t = sub.add_parser("check-threads", help="measure CPU time / wall time of a loaded model")
    t.add_argument("--model", required=True)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--n", type=int, default=20)
    t.add_argument("--threads", type=int, default=None)
    t.add_argument("--uncapped", action="store_true", help="stock Ultralytics (no ONNX Runtime / OpenVINO cap)")
    t.add_argument("--out", help="JSON file to write the measurement to")
    sub.add_parser("run-capped", help="run.py with the thread cap: python quantize.py run-capped <run.py args>")
    args = parser.parse_args(argv)
    if load_settings()["below_normal_priority"]:
        lower_priority()
    try:
        if args.cmd == "calib":
            build_calibration(force=args.force)
        elif args.cmd == "export":
            export(args.weights, args.fmt, args.int8, args.imgsz, force=args.force)
        elif args.cmd == "check-threads":
            res = verify_thread_cap(args.model, args.threads, args.imgsz, args.n, capped=not args.uncapped)
            print(json.dumps(res, indent=1))
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
