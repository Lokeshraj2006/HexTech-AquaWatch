"""Conservative appearance Re-ID CANDIDATES across cameras (AquaWatch).

    python reid.py --cameras samples/multicam/cameras.yaml --runs-root outputs/multicam --out reid_candidates.json

For every camera of the registry, the run folder is <runs-root>/<camera id> (as written by ingest.py):

  1. crops   : up to `max_crops_per_track` boxes per local track from stage1_tracking.jsonl (big enough, and
               preferably not overlapping another box), cut from the camera's own video with cv2 at the
               torso band of the box (boxes are in processing pixels and are scaled to the video size);
  2. embedding: HSV colour histogram (square-rooted) + mean Lab colour, L2-normalised; the track embedding
               is the L2-normalised mean of its crops. No learned model, nothing downloaded;
  3. matching: cosine similarity between tracks. Two tracks of the SAME camera that overlap in time are never
               matched (one person cannot be two tracks at once). Every candidate carries its time gap / overlap
               on the shared timeline when both cameras are aligned (cameras.normalized_time); without an
               alignment these are None and the candidate says "Temporal alignment unavailable.". With
               `time_window_s` set, aligned candidates further apart in time are dropped (unaligned: no gating);
  4. output  : per track the top_k candidates from other cameras, the best one of every other camera, and up to
               top_k from the same camera, ranked by score. Every link is "Candidate match - unverified" (status "unverified") with its score;
               global_id stays None. Colour similarity is weak evidence (same swim caps, water glare).

Embeddings are cached per run folder in reid_embeddings.json (keyed by the stage1 file hash, the video and the
embedding settings). score_candidates() measures top-1 / top-3 accuracy against a ground-truth mapping
(camera, track) -> person; ground truth is used ONLY for scoring, never as output.

cv2 is imported inside functions only (reading video frames and converting colours).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import cameras as camreg
import identity

# Thresholds. config.yaml / utils.DEFAULT_CONFIG may not be edited in this task, so the defaults live here;
# an optional `reid:` section of config.yaml overrides them (load_settings), CLI flags override both.
REID_DEFAULTS: dict[str, Any] = {
    "max_crops_per_track": 8,      # crops sampled per track (evenly over its lifetime)
    "min_box_h_px": 20.0,          # processing pixels; smaller boxes are too small for colour
    "min_det_conf": 0.0,           # skip boxes below this detector confidence
    "max_overlap_iou": 0.3,        # prefer boxes overlapping no other box of the frame more than this
    "torso_top": 0.15,             # torso band of the box, as fractions of its height / width
    "torso_bottom": 0.60,
    "torso_left": 0.20,
    "torso_right": 0.80,
    "min_crop_px": 6,              # minimum crop side in video pixels
    "hsv_bins": [16, 4, 4],        # H, S, V histogram bins
    "lab_weight": 0.5,             # weight of the 3 mean-Lab values against the histogram
    "top_k": 5,                    # candidates kept per track (other cameras; same camera separately)
    "min_score": 0.0,              # candidates below this cosine similarity are dropped
    "time_window_s": None,         # gate by normalized time (only when both cameras are aligned)
    "include_same_camera": True,   # also rank same-camera tracks that do NOT overlap in time
}
EMBED_KEYS = ("max_crops_per_track", "min_box_h_px", "min_det_conf", "max_overlap_iou", "torso_top",
              "torso_bottom", "torso_left", "torso_right", "min_crop_px", "hsv_bins", "lab_weight")
CACHE_NAME = "reid_embeddings.json"
CACHE_VERSION = 1
METHOD = "HSV histogram (sqrt) + mean Lab colour of torso crops, L2-normalised; cosine similarity; no learned model"
METHOD_TAG = "appearance_reid"         # short method name stored on every candidate link
DISCLAIMER = ("Appearance candidates only: every link is 'Candidate match - unverified'. Colour similarity is weak "
              "evidence; global_id stays null until a person verifies a link.")


class ReidError(RuntimeError):
    """A camera's run folder or video cannot be used for Re-ID."""


def load_settings(overrides: Mapping[str, Any] | None = None, use_config: bool = True) -> dict:
    """REID_DEFAULTS, then config.yaml's optional `reid:` section, then the non-None overrides."""
    settings = dict(REID_DEFAULTS)
    if use_config:
        try:
            import utils                                      # read-only; config.yaml has no reid section yet
            settings.update(utils.load_config().get("reid") or {})
        except Exception:                                     # a broken config must not stop Re-ID
            pass
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return settings


# ---------------------------------------------------------------------------- geometry

def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def torso_box(bbox: Iterable[float], settings: Mapping[str, Any]) -> list[float]:
    """The torso band of a person box (same coordinates): fractions torso_top..bottom, torso_left..right."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return [x1 + settings["torso_left"] * w, y1 + settings["torso_top"] * h,
            x1 + settings["torso_right"] * w, y1 + settings["torso_bottom"] * h]


def select_samples(track_rows: list[dict], rows_by_frame: Mapping[int, list[dict]],
                   settings: Mapping[str, Any]) -> list[dict]:
    """Up to max_crops_per_track observations of one track, evenly spread in time.

    Boxes that are too small or below min_det_conf are dropped; boxes overlapping another box of the same
    frame by more than max_overlap_iou are used only when the track has no clean box at all.
    """
    usable = [r for r in track_rows
              if r["frame"] is not None and (r["bbox"][3] - r["bbox"][1]) >= settings["min_box_h_px"]
              and (r["conf"] is None or r["conf"] >= settings["min_det_conf"])]
    clean = []
    for row in usable:
        others = [o for o in rows_by_frame.get(row["frame"], []) if o["track_id"] != row["track_id"]]
        if max((box_iou(row["bbox"], o["bbox"]) for o in others), default=0.0) <= settings["max_overlap_iou"]:
            clean.append(row)
    pool = sorted(clean or usable, key=lambda r: r["t"])
    n = int(settings["max_crops_per_track"])
    if n <= 0 or not pool:
        return []
    if len(pool) <= n:
        return pool
    picks = sorted({round(i * (len(pool) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [len(pool) // 2]
    return [pool[i] for i in picks]


# ---------------------------------------------------------------------------- embedding

def embed_crop(crop_bgr: np.ndarray, settings: Mapping[str, Any]) -> np.ndarray | None:
    """HSV histogram (square-rooted, L2) + weighted mean Lab colour of one BGR crop, L2-normalised (None if too small)."""
    import cv2
    if crop_bgr is None or crop_bgr.ndim != 3 or min(crop_bgr.shape[:2]) < int(settings["min_crop_px"]):
        return None
    hb, sb, vb = (int(v) for v in settings["hsv_bins"])
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)                # H 0..179, S and V 0..255
    h = hsv[..., 0].astype(np.int64) * hb // 180
    s = hsv[..., 1].astype(np.int64) * sb // 256
    v = hsv[..., 2].astype(np.int64) * vb // 256
    hist = np.bincount(((h * sb + s) * vb + v).ravel(), minlength=hb * sb * vb).astype(np.float64)
    hist = np.sqrt(hist / hist.sum())                              # Hellinger: sqrt of the distribution
    hist /= np.linalg.norm(hist) or 1.0
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64).mean(axis=0)
    lab_vec = np.array([lab[0] / 255.0, (lab[1] - 128.0) / 128.0, (lab[2] - 128.0) / 128.0])
    vector = np.concatenate([hist, float(settings["lab_weight"]) * lab_vec])
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else None


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray | None:
    """L2-normalised mean of crop embeddings (None for no crops)."""
    if not vectors:
        return None
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else None


def extract_track_embeddings(video_path: str | Path, meta: Mapping[str, Any], rows: list[dict],
                             settings: Mapping[str, Any]) -> dict:
    """Read the video once and embed every track. Returns {"tracks": {id: info}, "video_size", "scale", "notes"}.

    Raises ReidError when the video cannot be opened.
    """
    import cv2
    by_track: dict[int, list[dict]] = {}
    by_frame: dict[int, list[dict]] = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)
        if row["frame"] is not None:
            by_frame.setdefault(row["frame"], []).append(row)
    wanted: dict[int, list[dict]] = {}
    for track_id, track_rows in by_track.items():
        for row in select_samples(track_rows, by_frame, settings):
            wanted.setdefault(row["frame"], []).append(row)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ReidError(f"cannot open video {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    notes = []
    proc = meta.get("proc_size") or []
    if len(proc) == 2 and proc[0] and proc[1]:
        sx, sy = width / float(proc[0]), height / float(proc[1])
    else:
        sx = sy = 1.0
        notes.append("run has no proc_size; boxes assumed to be in video pixels")
    crops: dict[int, list[np.ndarray]] = {}
    used_frames: dict[int, list[int]] = {}
    last = max(wanted) if wanted else -1
    frame_no = 0
    try:
        while frame_no <= last:
            if frame_no in wanted:
                ok, frame = capture.read()
                if not ok:
                    notes.append(f"video ended at frame {frame_no} before frame {last}")
                    break
                for row in wanted[frame_no]:
                    x1, y1, x2, y2 = torso_box([row["bbox"][0] * sx, row["bbox"][1] * sy,
                                                row["bbox"][2] * sx, row["bbox"][3] * sy], settings)
                    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
                    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
                    vector = embed_crop(frame[y1:y2, x1:x2], settings) if x2 > x1 and y2 > y1 else None
                    if vector is not None:
                        crops.setdefault(row["track_id"], []).append(vector)
                        used_frames.setdefault(row["track_id"], []).append(frame_no)
            elif not capture.grab():
                notes.append(f"video ended at frame {frame_no} before frame {last}")
                break
            frame_no += 1
    finally:
        capture.release()

    tracks = {}
    for track_id, track_rows in sorted(by_track.items()):
        times = [r["t"] for r in track_rows]
        vector = mean_embedding(crops.get(track_id, []))
        tracks[int(track_id)] = {
            "first_s": round(min(times), 3), "last_s": round(max(times), 3),
            "n_boxes": len(track_rows), "n_crops": len(crops.get(track_id, [])),
            "frames": used_frames.get(track_id, []),
            "embedding": [round(float(x), 6) for x in vector] if vector is not None else None,
        }
    return {"tracks": tracks, "video_size": [width, height], "scale": [round(sx, 4), round(sy, 4)], "notes": notes}


# ---------------------------------------------------------------------------- per-run cache

def _file_sha1(path: Path) -> str:
    """SHA-1 of a file's bytes (identifies the tracking result the embeddings belong to)."""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_key(run_dir: Path, video_path: Path, settings: Mapping[str, Any]) -> dict:
    """What a cached embedding file must match: format version, tracking file hash, video, embedding settings."""
    stage1 = Path(run_dir) / "stage1_tracking.jsonl"
    video = Path(video_path)
    return {"version": CACHE_VERSION, "stage1_sha1": _file_sha1(stage1) if stage1.exists() else None,
            "video": video.name, "video_bytes": video.stat().st_size if video.exists() else None,
            "settings": {key: settings[key] for key in EMBED_KEYS}}


def load_cache(run_dir: Path, key: Mapping[str, Any]) -> dict | None:
    """The cached embeddings of a run folder when their key matches, else None."""
    path = Path(run_dir) / CACHE_NAME
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if document.get("key") != json.loads(json.dumps(key)):
        return None
    document["tracks"] = {int(k): v for k, v in (document.get("tracks") or {}).items()}
    return document


def save_cache(run_dir: Path, key: Mapping[str, Any], embedded: Mapping[str, Any]) -> Path:
    """Write reid_embeddings.json into the run folder."""
    path = Path(run_dir) / CACHE_NAME
    document = {"key": key, "method": METHOD, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "video_size": embedded.get("video_size"), "scale": embedded.get("scale"),
                "notes": embedded.get("notes", []),
                "tracks": {str(k): v for k, v in embedded["tracks"].items()}}
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def embed_run(camera: Mapping[str, Any], run_dir: str | Path, settings: Mapping[str, Any],
              use_cache: bool = True) -> dict:
    """Embeddings of every track of one camera's run (cached). Raises ReidError for a missing run / video."""
    run_dir = Path(run_dir)
    meta, rows, source = identity.read_observations(run_dir)
    if source != "stage1_tracking.jsonl":
        raise ReidError(f"{run_dir} has no stage1_tracking.jsonl")
    video = Path(camera.get("video_path") or camreg.resolve_path(camera.get("video") or meta.get("video") or ""))
    if not video.is_file():
        raise ReidError(f"video {video} not found")
    key = cache_key(run_dir, video, settings)
    cached = load_cache(run_dir, key) if use_cache else None
    if cached is not None:
        embedded, cache_state = cached, "hit"
    else:
        embedded = extract_track_embeddings(video, meta, rows, settings)
        cache_state = "miss" if use_cache else "off"
        if use_cache:
            save_cache(run_dir, key, embedded)
    return {"camera_id": str(camera["id"]), "run_dir": identity.project_path(run_dir),
            "video": identity.project_path(video), "cache": cache_state, "tracks": embedded["tracks"],
            "notes": embedded.get("notes", [])}


# ---------------------------------------------------------------------------- matching

def _overlap(a_first: float, a_last: float, b_first: float, b_last: float) -> bool:
    """True when two time spans share at least one instant."""
    return a_first <= b_last and b_first <= a_last


def _timing(item: Mapping[str, Any], other: Mapping[str, Any], cams: Mapping[str, dict]) -> tuple:
    """(gap_s, overlap_s, note) between two tracks' time spans.

    Same camera: raw video seconds. Other camera: the shared timeline (cameras.normalized_time); when either
    camera has no alignment the result is (None, None, "Temporal alignment unavailable.").
    """
    if item["camera_id"] == other["camera_id"]:
        spans = (item["first_s"], item["last_s"], other["first_s"], other["last_s"])
    else:
        a0, _ = camreg.normalized_time(cams.get(item["camera_id"], {}), item["first_s"])
        a1, _ = camreg.normalized_time(cams.get(item["camera_id"], {}), item["last_s"])
        b0, _ = camreg.normalized_time(cams.get(other["camera_id"], {}), other["first_s"])
        b1, _ = camreg.normalized_time(cams.get(other["camera_id"], {}), other["last_s"])
        if None in (a0, a1, b0, b1):
            return None, None, camreg.TEMPORAL_UNAVAILABLE
        spans = (a0, a1, b0, b1)
    later_start, earlier_end = max(spans[0], spans[2]), min(spans[1], spans[3])
    return round(max(0.0, later_start - earlier_end), 3), round(max(0.0, earlier_end - later_start), 3), None


def _rank_key(candidate: Mapping[str, Any]) -> tuple:
    """Sort order of candidates: best score first, then camera and track id (deterministic ties)."""
    return -candidate["score"], candidate["camera_id"], candidate["track_id"]


def rank_candidates(runs: list[dict], camera_list: list[dict], settings: Mapping[str, Any]) -> list[dict]:
    """Ranked, unverified candidate links for every track of `runs` (embed_run outputs).

    Never links two overlapping tracks of the same camera. Keeps the top_k candidates from OTHER cameras, plus
    the best candidate of every other camera (so one camera never pushes another camera out), plus up to top_k
    from the same camera (same-camera fragments never push cross-camera candidates out), merged and ranked by
    score. Each candidate carries time_gap_s / overlap_s; across cameras these need an
    alignment, else they are None and temporal_note says "Temporal alignment unavailable.". With
    time_window_s, candidates further apart in time are dropped (time_gate "applied"); unaligned pairs are
    never gated (time_gate "not_applied"). Returns one dict per track:
    {camera_id, track_id, display, global_id: None, verification_status, n_crops, first_s, last_s, candidates}.
    """
    cams = {str(c["id"]): c for c in camera_list}
    items = []
    for run in runs:
        for track_id, info in sorted(run["tracks"].items()):
            items.append({"camera_id": run["camera_id"], "track_id": int(track_id), **info})
    embedded = [i for i, item in enumerate(items) if item.get("embedding")]
    matrix = np.array([items[i]["embedding"] for i in embedded], dtype=np.float64) if embedded else np.zeros((0, 1))
    scores = matrix @ matrix.T if embedded else np.zeros((0, 0))
    position = {index: row for row, index in enumerate(embedded)}
    window = settings.get("time_window_s")
    top_k = int(settings.get("top_k", 5))

    results = []
    for i, item in enumerate(items):
        entry = {"camera_id": item["camera_id"], "track_id": item["track_id"],
                 "display": identity.display_track(item["track_id"], item["camera_id"]), "global_id": None,
                 "n_crops": item.get("n_crops", 0), "first_s": item["first_s"], "last_s": item["last_s"],
                 "candidates": []}
        if i not in position:
            entry["verification_status"] = identity.UNAVAILABLE
            entry["note"] = "no usable crops, no appearance embedding"
            results.append(entry)
            continue
        found = []
        for j in embedded:
            if j == i:
                continue
            other = items[j]
            same_camera = other["camera_id"] == item["camera_id"]
            if same_camera and (not settings.get("include_same_camera", True)
                                or _overlap(item["first_s"], item["last_s"], other["first_s"], other["last_s"])):
                continue                                        # one person is never two tracks at once
            gap, overlap, note = _timing(item, other, cams)
            gate = "off"
            if window is not None:
                if gap is None:
                    gate = "not_applied"                        # no alignment: times are not comparable
                elif gap > float(window):
                    continue
                else:
                    gate = "applied"
            score = float(scores[position[i], position[j]])
            if score < float(settings.get("min_score", 0.0)):
                continue
            found.append({"camera_id": other["camera_id"], "track_id": other["track_id"],
                          "display": identity.display_track(other["track_id"], other["camera_id"]),
                          "score": round(score, 4), "status": identity.UNVERIFIED, "label": identity.CANDIDATE_LABEL,
                          "method": METHOD_TAG, "same_camera": same_camera, "time_gap_s": gap, "overlap_s": overlap,
                          "time_gate": gate, "temporal_note": note})
        cross = sorted((c for c in found if not c["same_camera"]), key=_rank_key)
        best_per_camera = {}
        for candidate in cross:                                 # sorted: the first one per camera is its best
            best_per_camera.setdefault(candidate["camera_id"], candidate)
        kept = cross[:top_k] + [c for c in best_per_camera.values() if c not in cross[:top_k]]
        same = sorted((c for c in found if c["same_camera"]), key=_rank_key)[:top_k]
        found = sorted(kept + same, key=_rank_key)
        for rank, candidate in enumerate(found, start=1):
            candidate["rank"] = rank
        entry["candidates"] = found
        entry["verification_status"] = identity.status_for_candidates(found)
        results.append(entry)
    return results


def run_reid(camera_list: list[dict], runs_root: str | Path, settings: Mapping[str, Any] | None = None,
             use_cache: bool = True, echo=print, dataset_label: str | None = None) -> dict:
    """Embed every camera's run (<runs_root>/<id>) and rank candidates. A broken camera is reported and skipped.

    dataset_label (cameras.registry_label, e.g. "VIRTUAL MULTI-CAMERA DEMO - ...") is copied into the result.
    """
    settings = load_settings(settings)
    runs_root = camreg.resolve_path(runs_root)
    runs, camera_rows = [], []
    for camera in camera_list:
        run_dir = runs_root / str(camera["id"])
        info = camreg.sync_info(camera)
        row = {"camera_id": str(camera["id"]), "run_dir": identity.project_path(run_dir), **info,
               "n_tracks": 0, "n_embedded": 0, "cache": None, "error": None}
        started = time.time()
        try:
            run = embed_run(camera, run_dir, settings, use_cache=use_cache)
        except Exception as exc:                                  # one bad camera must not stop the others
            row["error"] = f"{type(exc).__name__}: {exc}"
            echo(f"  {camera['id']}: SKIPPED ({row['error']})".encode("ascii", "replace").decode("ascii"))
            camera_rows.append(row)
            continue
        runs.append(run)
        row.update(n_tracks=len(run["tracks"]), cache=run["cache"], notes=run["notes"],
                   n_embedded=sum(1 for t in run["tracks"].values() if t.get("embedding")))
        echo(f"  {camera['id']}: {row['n_embedded']}/{row['n_tracks']} tracks embedded "
             f"(cache {run['cache']}, {time.time() - started:.1f} s, sync {info['sync_status']})")
        camera_rows.append(row)
    tracks = rank_candidates(runs, camera_list, settings)
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "dataset_label": dataset_label, "method": METHOD,
            "disclaimer": DISCLAIMER, "settings": {k: settings[k] for k in REID_DEFAULTS}, "cameras": camera_rows,
            "tracks": tracks}


# ---------------------------------------------------------------------------- scoring against ground truth

def load_ground_truth(path: str | Path) -> dict[tuple[str, int], str]:
    """Read a (camera, track) -> person mapping (FOR SCORING ONLY).

    Accepts a JSON list [{"camera_id", "track_id", "person"}] or a dict {"cam_a:4": "P1", ...}.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mapping: dict[tuple[str, int], str] = {}
    if isinstance(data, dict):
        for key, person in data.items():
            camera_id, _, track = str(key).rpartition(":")
            mapping[(camera_id, int(track))] = str(person)
    else:
        for row in data:
            mapping[(str(row["camera_id"]), int(row["track_id"]))] = str(row["person"])
    return mapping


def score_candidates(result: Mapping[str, Any], ground_truth: Mapping[tuple[str, int], Any],
                     cross_camera_only: bool = True) -> dict:
    """Top-1 / top-3 candidate accuracy of a run_reid() result against a (camera, track) -> person mapping.

    A query counts when its track has a ground-truth person AND another eligible track of the same person
    exists (another camera when cross_camera_only, never an overlapping track of the same camera). A query
    without any candidate counts as a miss. Ground truth is used for scoring only.
    """
    tracks = list(result.get("tracks") or [])
    n = top1 = top3 = no_gt = no_match = 0
    misses = []
    for track in tracks:
        key = (str(track["camera_id"]), int(track["track_id"]))
        person = ground_truth.get(key)
        if person is None:
            no_gt += 1
            continue
        true_set = set()
        for other in tracks:
            other_key = (str(other["camera_id"]), int(other["track_id"]))
            if other_key == key or ground_truth.get(other_key) != person:
                continue
            same = other_key[0] == key[0]
            if same and (cross_camera_only or _overlap(track["first_s"], track["last_s"],
                                                       other["first_s"], other["last_s"])):
                continue
            true_set.add(other_key)
        if not true_set:
            no_match += 1
            continue
        ranked = [(str(c["camera_id"]), int(c["track_id"])) for c in track.get("candidates") or []
                  if not (cross_camera_only and c.get("same_camera"))]
        n += 1
        hit1 = bool(ranked[:1]) and ranked[0] in true_set
        hit3 = any(c in true_set for c in ranked[:3])
        top1 += hit1
        top3 += hit3
        if not hit1:
            misses.append({"query": list(key), "true": sorted(list(k) for k in true_set),
                           "ranked": [list(c) for c in ranked[:3]]})
    return {"n_queries": n, "top1_hits": top1, "top3_hits": top3,
            "top1_accuracy": round(top1 / n, 4) if n else None, "top3_accuracy": round(top3 / n, 4) if n else None,
            "skipped_no_ground_truth": no_gt, "skipped_no_true_match": no_match, "cross_camera_only": cross_camera_only,
            "top1_misses": misses, "note": "Ground truth used for scoring only; never as CV output."}


# ---------------------------------------------------------------------------- CLI

def summary_lines(result: Mapping[str, Any], limit: int = 10) -> list[str]:
    """Plain ASCII lines: per-camera embedding counts, then the best candidate of the first tracks."""
    lines = [f"Re-ID candidates ({result['method']})", result["disclaimer"]]
    if result.get("dataset_label"):
        lines.insert(0, f"DATASET: {result['dataset_label']}")
    for camera in result["cameras"]:
        status = f"ERROR {camera['error']}" if camera.get("error") else \
            f"{camera['n_embedded']}/{camera['n_tracks']} tracks embedded, cache {camera['cache']}"
        lines.append(f"  {camera['camera_id']:<14} sync {camera['sync_status']:<13} {status}")
    shown = 0
    for track in result["tracks"]:
        if shown >= limit:
            lines.append(f"  ... {len(result['tracks']) - limit} more tracks in the output file")
            break
        if track["candidates"]:
            best = track["candidates"][0]
            note = f" [{best['temporal_note']}]" if best.get("temporal_note") else ""
            lines.append(f"  {track['display']} -> {identity.candidate_text(best)}{note}")
        else:
            lines.append(f"  {track['display']} -> {track.get('note') or identity.NO_IDENTITY_EVIDENCE}")
        shown += 1
    return [line.encode("ascii", "replace").decode("ascii") for line in lines]


def main(argv=None) -> int:
    """CLI: embed every camera's run, rank candidates, write JSON. Exit 0 ok, 1 some camera failed, 2 bad input."""
    parser = argparse.ArgumentParser(description="AquaWatch: unverified appearance Re-ID candidates across cameras")
    parser.add_argument("--cameras", required=True, metavar="cameras.yaml", help="camera registry")
    parser.add_argument("--runs-root", required=True, metavar="DIR", help="run folders are <runs-root>/<camera id>")
    parser.add_argument("--out", required=True, metavar="JSON", help="where to write the candidate list")
    parser.add_argument("--only", nargs="+", metavar="ID", help="use only these camera ids")
    parser.add_argument("--top-k", type=int, help=f"candidates per track from other cameras, and again from the same camera "
                             f"(default {REID_DEFAULTS['top_k']})")
    parser.add_argument("--max-crops", type=int, dest="max_crops_per_track",
                        help=f"crops per track (default {REID_DEFAULTS['max_crops_per_track']})")
    parser.add_argument("--min-score", type=float, help="drop candidates below this cosine similarity")
    parser.add_argument("--time-window", type=float, dest="time_window_s",
                        help="gate candidates by normalized time (only for aligned cameras)")
    parser.add_argument("--no-same-camera", action="store_true", help="rank only tracks of other cameras")
    parser.add_argument("--no-cache", action="store_true", help=f"ignore and do not write {CACHE_NAME}")
    parser.add_argument("--ground-truth", metavar="JSON", help="score top-1/top-3 against (camera, track) -> person")
    args = parser.parse_args(argv)

    try:
        registry = camreg.load_registry(args.cameras, quiet=True)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    camera_list = registry["cameras"]
    if args.only:
        camera_list = [c for c in camera_list if c["id"] in set(args.only)]
        if not camera_list:
            print("ERROR: none of the --only camera ids is in the registry")
            return 2
    overrides = {"top_k": args.top_k, "max_crops_per_track": args.max_crops_per_track, "min_score": args.min_score,
                 "time_window_s": args.time_window_s}
    if args.no_same_camera:
        overrides["include_same_camera"] = False
    print(f"AquaWatch Re-ID: {len(camera_list)} camera(s), runs in {args.runs_root}")
    if registry.get("label"):
        print(f"DATASET: {registry['label']}".encode("ascii", "replace").decode("ascii"))
    result = run_reid(camera_list, args.runs_root, overrides, use_cache=not args.no_cache,
                      dataset_label=registry.get("label"))
    result["site"] = registry.get("site") or None
    if args.ground_truth:
        result["scoring"] = score_candidates(result, load_ground_truth(args.ground_truth))
    out = camreg.resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for line in summary_lines(result)[1 if result.get("dataset_label") else 0:]:   # label already printed
        print(line)
    if "scoring" in result:
        s = result["scoring"]
        print(f"scoring (ground truth for scoring only): {s['n_queries']} queries, top-1 {s['top1_accuracy']}, "
              f"top-3 {s['top3_accuracy']}")
    print(f"wrote {out}")
    return 1 if any(c.get("error") for c in result["cameras"]) else 0


if __name__ == "__main__":
    sys.exit(main())
