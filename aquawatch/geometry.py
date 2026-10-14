"""Cross-camera track links from KNOWN camera geometry (AquaWatch).

    python geometry.py --cameras data/virtual_cctv/cameras.yaml --geometry data/virtual_cctv/geometry.json \
        --runs-root outputs/virtual_cctv --out outputs/virtual_cctv/geometry_links.json

When the geometry between two overlapping cameras is known exactly, a swimmer seen by both cameras at the same
moment must have (almost) the same box once both boxes are mapped into one shared pixel space. Here the geometry is
exact because the demo cameras are crops of ONE recording (data/virtual_cctv/geometry.json: crop_xyxy of each view
in source pixels). At a real site the same role is played by a measured calibration / homography between
overlapping cameras plus synchronised clocks.

  1. map   : every stage1_tracking.jsonl box of a camera run (processing pixels, meta.proc_size) is scaled by
             crop_size / proc_size, shifted by the crop offset into SOURCE pixels and clipped to the camera's crop.
             A box touching the crop border (within edge_margin_px) is "cut" by the crop and is not compared;
  2. compare: for every pair of cameras whose crops overlap (and whose clocks are aligned: both "synchronized",
             same offset, same fps / stride), for every pair of tracks seen on the same frames: IoU of the two
             source boxes on every shared frame. A pair is linked when it has >= min_shared_frames compared frames
             and median IoU >= min_iou. A track links to at most one track per other camera (the best score wins);
             the losers are reported as conflicts;
  3. ids   : connected components of the links get global ids "G001", ... ONLY for tracks with a link. A component
             holding two tracks of one camera seen at the same moment is contradictory: no global id, status
             "conflict". Tracks without a link keep global_id null, status "unavailable".

Status "verified_automated" means "certain from the known camera geometry, given the detector boxes in both views";
it is NOT a check by a person and it is not validated against human annotation (none exists for the demo data).
Tracker IDs are local to one camera's run ("Track #N (cam02)"), never a real identity.

Optionally compares the links with reid.py's appearance candidates (reid_candidates.json) and writes the
"agreement with geometry links" (never called accuracy: geometry links are not ground truth).

Pure Python + PyYAML (through cameras.py); no cv2 / numpy / torch.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import cameras as camreg
import identity

# Thresholds: function defaults + CLI flags (config.yaml / utils.py may not be edited in this task).
DEFAULTS: dict[str, Any] = {
    "min_shared_frames": 10,   # compared frames (both boxes present, neither cut by a crop edge) needed for a link
    "min_iou": 0.5,            # median IoU of the source-pixel boxes over the compared frames
    "edge_margin_px": 2.0,     # a box within this many SOURCE pixels of its crop border is cut by the crop
    "fps_tolerance": 1e-3,     # relative fps difference allowed between two runs compared frame by frame
    "merge_gap_s": 1.0,        # agreeing moments closer than this form one agreement span (evidence)
    "event_min_frames": 5,     # compared frames inside an event window needed to name its swimmer in another view
    "reid_top_n": 3,           # "in the top-N appearance candidates" for the Re-ID comparison
    "n_examples": 3,           # examples per camera pair in the summary
}
METHOD = "known_camera_geometry"
LINK_STATUS = "verified_automated"   # same value as identity.VERIFIED_AUTOMATED
NO_LINK_STATUS = "unavailable"       # same value as identity.UNAVAILABLE
CONFLICT_STATUS = "conflict"
STAGE1 = "stage1_tracking.jsonl"
POOL_BEHAVIOURS = ("aquatic_warning", "aquatic_distress", "submersion", "possible_submersion")
CAVEATS = [
    "The links rely on the detector boxes in both views: a swimmer the detector misses in one view gets no link, "
    "and two swimmers whose boxes merge in both views can be linked wrongly.",
    "The views are crops of ONE recording: they share lighting, clock and viewpoint, so linking them is far easier "
    "than real cross-camera Re-ID. At a real site this needs a measured calibration / homography between "
    "overlapping cameras and synchronised clocks.",
    "Not validated against human annotation: no ground truth exists for this footage. 'verified_automated' means "
    "certain from the known crop geometry given the detector boxes, not checked by a person.",
    "Track IDs are the stitched tracker IDs of each camera's run (stage1_tracking.jsonl). A tracker ID is not a "
    "real identity, and a wrong stitch in one view carries its link to another swimmer.",
    "Boxes touching a crop border are cut by the crop and are skipped for IoU; swimmers that stay near a crop edge "
    "get fewer compared frames and may stay unlinked.",
    "A track links to at most one track per other camera; extra matches are listed under 'conflicts' (often one "
    "swimmer split into two tracks by the tracker of one view).",
    "Tracks without any geometry link keep global_id null (status 'unavailable').",
]


class GeometryError(ValueError):
    """The geometry file or a run folder cannot be used."""


def display(camera_id: Any, track_id: Any) -> str:
    """'Track #4 (cam02)': a local tracker ID of one camera, never 'person 4'."""
    return f"Track #{int(track_id)} ({camera_id})"


def _ascii(text: str) -> str:
    """Plain ASCII for Windows consoles."""
    return str(text).encode("ascii", "replace").decode("ascii")


# ---------------------------------------------------------------------------- geometry file + boxes

def load_geometry(path: str | Path) -> dict:
    """Read geometry.json -> {"path", "label", "kind", "source_size", "sync", "ground_truth", "cameras": {id: {...}}}.

    Every camera needs crop_xyxy [x0, y0, x1, y1] (source pixels, x1 > x0, y1 > y0); size defaults to the crop size
    and must agree with it.
    """
    path = Path(path)
    if not path.exists():
        raise GeometryError(f"geometry file {path} not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    cams: dict[str, dict] = {}
    for raw in data.get("cameras") or []:
        cam_id = str(raw.get("id") or "").strip()
        crop = raw.get("crop_xyxy")
        if not cam_id or not isinstance(crop, (list, tuple)) or len(crop) != 4:
            raise GeometryError(f"geometry camera {raw!r}: needs id and crop_xyxy [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in crop)
        if x1 <= x0 or y1 <= y0:
            raise GeometryError(f"geometry camera {cam_id}: empty crop {crop}")
        size = [float(v) for v in (raw.get("size") or [x1 - x0, y1 - y0])]
        if abs(size[0] - (x1 - x0)) > 1.0 or abs(size[1] - (y1 - y0)) > 1.0:
            raise GeometryError(f"geometry camera {cam_id}: size {size} does not match crop {crop}")
        cams[cam_id] = {"id": cam_id, "name": raw.get("name") or cam_id, "crop_xyxy": [x0, y0, x1, y1],
                        "size": size}
    if not cams:
        raise GeometryError(f"geometry file {path} lists no cameras")
    return {"path": path, "label": data.get("label"), "kind": data.get("kind"),
            "source_size": data.get("source_size"), "sync": data.get("sync"),
            "ground_truth": data.get("ground_truth"), "cameras": cams}


def crop_overlap(a: Iterable[float], b: Iterable[float]) -> list[float] | None:
    """Intersection [x0, y0, x1, y1] of two crops (source pixels), or None when they do not overlap."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    return [x0, y0, x1, y1] if x1 > x0 and y1 > y0 else None


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_source_box(bbox: Iterable[float], proc_size: Iterable[float], crop_xyxy: Iterable[float],
                  edge_margin_px: float = DEFAULTS["edge_margin_px"]) -> tuple[list[float] | None, bool]:
    """A box in processing pixels of one camera -> (box in SOURCE pixels clipped to the crop, cut_by_crop_edge).

    Scale per axis by crop_size / proc_size, add the crop offset, clip to the crop. A box within edge_margin_px of a
    crop border is cut (the crop hides part of the swimmer). Returns (None, True) when nothing is left after clipping.
    """
    x0, y0, x1, y1 = (float(v) for v in crop_xyxy)
    proc_w, proc_h = (float(v) for v in proc_size)
    if proc_w <= 0 or proc_h <= 0:
        raise GeometryError(f"bad proc_size {list(proc_size)}")
    sx, sy = (x1 - x0) / proc_w, (y1 - y0) / proc_h
    bx1, by1, bx2, by2 = (float(v) for v in bbox)
    box = [min(max(x0 + bx1 * sx, x0), x1), min(max(y0 + by1 * sy, y0), y1),
           min(max(x0 + bx2 * sx, x0), x1), min(max(y0 + by2 * sy, y0), y1)]
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        return None, True
    m = float(edge_margin_px)
    cut = box[0] <= x0 + m or box[1] <= y0 + m or box[2] >= x1 - m or box[3] >= y1 - m
    return [round(v, 2) for v in box], cut


def inside(box: Iterable[float], region: Iterable[float], margin: float = 0.0) -> bool:
    """True when the box lies fully inside the region shrunk by `margin` on every side."""
    bx1, by1, bx2, by2 = box
    rx0, ry0, rx1, ry1 = region
    return bx1 >= rx0 + margin and by1 >= ry0 + margin and bx2 <= rx1 - margin and by2 <= ry1 - margin


# ---------------------------------------------------------------------------- run folders

def read_stage1(run_dir: str | Path) -> tuple[dict, list[dict]]:
    """(meta, rows) of <run_dir>/stage1_tracking.jsonl; rows are {track_id, frame, t, bbox, conf}."""
    path = Path(run_dir) / STAGE1
    if not path.exists():
        raise GeometryError(f"{path} not found")
    meta, rows = {}, []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "meta" in row:
                meta = dict(row["meta"] or {})
                continue
            if row.get("person_id") is None or row.get("frame") is None or not row.get("bbox"):
                continue
            rows.append({"track_id": int(row["person_id"]), "frame": int(row["frame"]),
                         "t": float(row["timestamp"]), "bbox": [float(v) for v in row["bbox"]],
                         "conf": float(row["conf"]) if row.get("conf") is not None else None})
    if not meta.get("proc_size"):
        raise GeometryError(f"{path}: meta line without proc_size")
    return meta, rows


def map_observations(camera_id: str, meta: Mapping[str, Any], rows: list[dict], crop_xyxy: Iterable[float],
                     edge_margin_px: float = DEFAULTS["edge_margin_px"]) -> list[dict]:
    """Every row of a run -> {camera_id, track_id, frame, t, box (source pixels), cut}. Boxes clipped away are dropped."""
    observations = []
    for row in rows:
        box, cut = to_source_box(row["bbox"], meta["proc_size"], crop_xyxy, edge_margin_px)
        if box is None:
            continue
        observations.append({"camera_id": camera_id, "track_id": row["track_id"], "frame": row["frame"],
                             "t": row["t"], "box": box, "cut": cut})
    return observations


def alignment(camera_a: Mapping[str, Any], camera_b: Mapping[str, Any], meta_a: Mapping[str, Any],
              meta_b: Mapping[str, Any], fps_tolerance: float = DEFAULTS["fps_tolerance"]) -> tuple[bool, str]:
    """Can two runs be compared frame index by frame index? (ok, reason)

    Needs both cameras "synchronized" in the registry with the same offset_s, and the same fps and frame stride.
    """
    sync_a, sync_b = camreg.sync_info(dict(camera_a)), camreg.sync_info(dict(camera_b))
    for cam, info in ((camera_a, sync_a), (camera_b, sync_b)):
        if info["sync_status"] != camreg.SYNC_SYNCHRONIZED:
            return False, f"{cam['id']} sync_status is {info['sync_status']}: {camreg.TEMPORAL_UNAVAILABLE}"
    if abs(float(sync_a["offset_s"] or 0.0) - float(sync_b["offset_s"] or 0.0)) > 1e-9:
        return False, (f"offsets differ ({sync_a['offset_s']} s vs {sync_b['offset_s']} s): frame indices do not "
                       "refer to the same moment")
    fps_a, fps_b = float(meta_a.get("fps") or 0.0), float(meta_b.get("fps") or 0.0)
    if fps_a <= 0 or fps_b <= 0 or abs(fps_a - fps_b) > fps_tolerance * max(fps_a, fps_b):
        return False, f"fps differ ({fps_a} vs {fps_b})"
    if int(meta_a.get("stride") or 1) != int(meta_b.get("stride") or 1):
        return False, f"frame strides differ ({meta_a.get('stride')} vs {meta_b.get('stride')})"
    return True, f"synchronized, offset {float(sync_a['offset_s'] or 0.0)} s, fps {fps_a}"


# ---------------------------------------------------------------------------- linking

def compare_tracks(obs_a: list[dict], obs_b: list[dict]) -> dict[tuple[int, int], dict]:
    """Per (track_a, track_b) seen on the same frames: frames both present, frames cut, IoUs of compared frames."""
    frames_b: dict[int, list[dict]] = {}
    for ob in obs_b:
        frames_b.setdefault(ob["frame"], []).append(ob)
    stats: dict[tuple[int, int], dict] = {}
    for oa in obs_a:
        for ob in frames_b.get(oa["frame"], ()):
            entry = stats.setdefault((oa["track_id"], ob["track_id"]),
                                     {"both": 0, "cut": 0, "ious": [], "frames": [], "times": []})
            entry["both"] += 1
            if oa["cut"] or ob["cut"]:
                entry["cut"] += 1
                continue
            entry["ious"].append(box_iou(oa["box"], ob["box"]))
            entry["frames"].append(oa["frame"])
            entry["times"].append(oa["t"])
    return stats


def spans(times: Iterable[float], merge_gap_s: float = DEFAULTS["merge_gap_s"]) -> list[list[float]]:
    """Sorted moments -> [[start_s, end_s], ...], joining moments closer than merge_gap_s."""
    out: list[list[float]] = []
    for t in sorted(times):
        if out and t - out[-1][1] <= merge_gap_s:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def pair_evidence(entry: Mapping[str, Any], min_iou: float, merge_gap_s: float = DEFAULTS["merge_gap_s"]) -> dict:
    """Evidence numbers of one track pair.

    shared_frames = compared frames (both boxes present, neither cut); median_iou over them; agree_frames = compared
    frames with IoU >= min_iou and agree_spans_s = when they happen (a track that switched swimmers agrees only
    part of the time).
    """
    ious = entry["ious"]
    n = len(ious)
    agree_times = [t for t, v in zip(entry["times"], ious) if v >= min_iou]
    return {"shared_frames": n, "frames_both_present": entry["both"], "frames_skipped_edge_cut": entry["cut"],
            "median_iou": round(statistics.median(ious), 4) if n else None,
            "agree_frames": len(agree_times),
            "frac_frames_iou_ge_min": round(len(agree_times) / n, 4) if n else None,
            "first_s": round(min(entry["times"]), 3) if n else None,
            "last_s": round(max(entry["times"]), 3) if n else None,
            "first_frame": min(entry["frames"]) if n else None, "last_frame": max(entry["frames"]) if n else None,
            "agree_spans_s": spans(agree_times, merge_gap_s)}


def candidate_links(camera_a: str, camera_b: str, stats: Mapping[tuple[int, int], dict],
                    min_shared_frames: int = DEFAULTS["min_shared_frames"], min_iou: float = DEFAULTS["min_iou"],
                    merge_gap_s: float = DEFAULTS["merge_gap_s"]) -> tuple[list[dict], list[dict]]:
    """(links passing both thresholds, every compared pair with its evidence) of one camera pair.

    A link carries "_agree_frames" (set of frames whose boxes agree) for conflict checks; build_links() drops it
    before writing JSON.
    """
    links, compared = [], []
    for (track_a, track_b), entry in sorted(stats.items()):
        evidence = pair_evidence(entry, min_iou, merge_gap_s)
        compared.append({"camera_a": camera_a, "track_a": track_a, "camera_b": camera_b, "track_b": track_b,
                         "evidence": evidence})
        if evidence["shared_frames"] >= int(min_shared_frames) and evidence["median_iou"] is not None \
                and evidence["median_iou"] >= float(min_iou):
            links.append({"camera_a": camera_a, "track_a": track_a, "camera_b": camera_b, "track_b": track_b,
                          "display_a": display(camera_a, track_a), "display_b": display(camera_b, track_b),
                          "status": LINK_STATUS, "method": METHOD, "evidence": evidence,
                          "score": evidence["median_iou"],
                          "_agree_frames": {f for f, v in zip(entry["frames"], entry["ious"]) if v >= min_iou}})
    return links, compared


def _rank_key(link: Mapping[str, Any]) -> tuple:
    """Best first: more frames on which the boxes agree, then higher median IoU (score), then a stable order."""
    ev = link["evidence"]
    return (-ev.get("agree_frames", 0), -link["score"], -ev["shared_frames"], link["camera_a"], link["track_a"],
            link["camera_b"], link["track_b"])


def resolve_one_to_one(candidates: list[dict], track_frames: Mapping[tuple[str, int], set] | None = None
                       ) -> tuple[list[dict], list[dict]]:
    """Keep at most one partner per (track, other camera), best first (_rank_key). Returns (links, conflicts).

    A conflict names the rejected link, the kept link that blocked it and its kind: "time_disjoint" when the two
    links agree at different moments (one swimmer split into two tracks, or a track that switched swimmers),
    "simultaneous" when they agree on the same frames (two boxes of one view on one box of the other).
    """
    track_frames = track_frames or {}
    partner: dict[tuple[str, int, str], dict] = {}
    kept, conflicts = [], []
    for link in sorted(candidates, key=_rank_key):
        key_a = (link["camera_a"], link["track_a"], link["camera_b"])
        key_b = (link["camera_b"], link["track_b"], link["camera_a"])
        blocker = partner.get(key_a) or partner.get(key_b)
        if blocker is None:
            partner[key_a] = partner[key_b] = link
            kept.append(link)
            continue
        if blocker["camera_a"] == link["camera_a"] and blocker["track_a"] == link["track_a"]:
            hub = (link["camera_a"], link["track_a"])
            camera, mine, theirs = link["camera_b"], link["track_b"], blocker["track_b"]
        else:
            hub = (link["camera_b"], link["track_b"])
            camera, mine, theirs = link["camera_a"], link["track_a"], blocker["track_a"]
        if "_agree_frames" in link and "_agree_frames" in blocker:
            shared = link["_agree_frames"] & blocker["_agree_frames"]
            basis = "frames on which both matches agree"
        else:
            shared = track_frames.get((camera, mine), set()) & track_frames.get((camera, theirs), set())
            basis = "frames where both competing tracks are present"
        kind = "simultaneous" if shared else "time_disjoint"
        note = (f"{display(*hub)} matches {display(camera, theirs)} (kept) and {display(camera, mine)}; "
                + (f"{len(shared)} {basis}, so only one of them can be this swimmer there."
                   if kind == "simultaneous" else "the matches happen at different moments: one swimmer split into "
                   "two tracks, or one track that switched swimmers."))
        conflicts.append({"rejected": _short(link), "kept": _short(blocker), "kind": kind,
                          "competing_tracks": [display(camera, theirs), display(camera, mine)],
                          "overlapping_frames": len(shared), "note": note})
    kept.sort(key=lambda l: (l["camera_a"], l["camera_b"], l["track_a"], l["track_b"]))
    return kept, conflicts


def _short(link: Mapping[str, Any]) -> dict:
    """A compact copy of a link for conflict lists."""
    ev = link["evidence"]
    return {"camera_a": link["camera_a"], "track_a": link["track_a"], "camera_b": link["camera_b"],
            "track_b": link["track_b"], "score": link["score"], "shared_frames": ev["shared_frames"],
            "agree_frames": ev.get("agree_frames")}


def _pair_key(u: tuple[str, int], v: tuple[str, int]) -> tuple:
    """Order-free key of two (camera, track) nodes."""
    return (u, v) if u <= v else (v, u)


def assign_global_ids(links: list[dict], track_frames: Mapping[tuple[str, int], set],
                      track_first: Mapping[tuple[str, int], float] | None = None,
                      pair_stats: Mapping[tuple, Mapping[str, Any]] | None = None,
                      min_shared_frames: int = DEFAULTS["min_shared_frames"],
                      min_iou: float = DEFAULTS["min_iou"]) -> tuple[dict, dict, list[dict]]:
    """Connected components of the links -> (global_ids {G001: [[cam, track], ...]}, {(cam, track): gid}, conflicts).

    A component is contradictory, gets no global id and is returned in the conflicts list when two of its tracks
      - belong to one camera and share a frame (one swimmer cannot be two tracks at once), or
      - belong to two compared cameras, have >= min_shared_frames compared frames and a median IoU < min_iou
        (`pair_stats`, evidence keyed by _pair_key): the geometry says they were in different places at once.
    """
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for link in links:
        a, b = (link["camera_a"], link["track_a"]), (link["camera_b"], link["track_b"])
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    track_first, pair_stats = track_first or {}, pair_stats or {}
    ordered = sorted((sorted(members) for members in groups.values()),
                     key=lambda m: (min(track_first.get(n, 0.0) for n in m), m[0]))
    global_ids, member_of, conflicts = {}, {}, []
    for members in ordered:
        clash = []
        for i, first in enumerate(members):
            for second in members[i + 1:]:
                if first[0] == second[0]:
                    shared = track_frames.get(first, set()) & track_frames.get(second, set())
                    if shared:
                        clash.append({"tracks": [display(*first), display(*second)], "kind": "same_camera_overlap",
                                      "shared_frames": len(shared)})
                    continue
                ev = pair_stats.get(_pair_key(first, second))
                if ev and ev["shared_frames"] >= int(min_shared_frames) and ev["median_iou"] is not None \
                        and ev["median_iou"] < float(min_iou):
                    clash.append({"tracks": [display(*first), display(*second)], "kind": "geometry_disagrees",
                                  "shared_frames": ev["shared_frames"], "median_iou": ev["median_iou"]})
        if clash:
            conflicts.append({"kind": "contradictory_component", "members": [list(m) for m in members],
                              "clashes": clash,
                              "note": "the links chain tracks that cannot be one swimmer (one camera, same moment; "
                                      "or boxes in different places at the same moment); at least one link is "
                                      "wrong, so no global id is given."})
            continue
        gid = f"G{len(global_ids) + 1:03d}"
        global_ids[gid] = [list(m) for m in members]
        for member in members:
            member_of[member] = gid
    return global_ids, member_of, conflicts


# ---------------------------------------------------------------------------- diagnostics

def pair_link_rate(camera_a: str, camera_b: str, region: Iterable[float], obs_a: list[dict], links: list[dict],
                   compared: list[dict], min_shared_frames: int, edge_margin_px: float,
                   rejected: Iterable[tuple] = ()) -> dict:
    """How often a track of camera_a that stays inside the camera_a/camera_b overlap gets a camera_b partner.

    Eligible: >= min_shared_frames uncut boxes lying fully inside the overlap region (shrunk by edge_margin_px).
    Unlinked eligible tracks list their best camera_b pair (highest median IoU) to show why.
    uncut_boxes_inside_overlap counts camera_a's boxes in the overlap (compare with the reverse direction to see
    which view's detector finds more swimmers there).
    """
    region = list(region)
    inside_count: dict[int, int] = {}
    for ob in obs_a:
        if not ob["cut"] and inside(ob["box"], region, edge_margin_px):
            inside_count[ob["track_id"]] = inside_count.get(ob["track_id"], 0) + 1
    eligible = sorted(t for t, n in inside_count.items() if n >= int(min_shared_frames))
    partner = {}
    for link in links:
        if link["camera_a"] == camera_a and link["camera_b"] == camera_b:
            partner[link["track_a"]] = link["track_b"]
        elif link["camera_b"] == camera_a and link["camera_a"] == camera_b:
            partner[link["track_b"]] = link["track_a"]
    best: dict[int, dict] = {}
    for pair in compared:
        if pair["camera_a"] == camera_a and pair["camera_b"] == camera_b:
            mine, other = pair["track_a"], pair["track_b"]
        elif pair["camera_b"] == camera_a and pair["camera_a"] == camera_b:
            mine, other = pair["track_b"], pair["track_a"]
        else:
            continue
        ev = pair["evidence"]
        if ev["median_iou"] is None:
            continue
        current = best.get(mine)
        if current is None or (ev["median_iou"], ev["shared_frames"]) > (current["median_iou"], current["shared_frames"]):
            best[mine] = {"track": other, "median_iou": ev["median_iou"], "shared_frames": ev["shared_frames"]}
    rejected = {tuple(r) for r in rejected}
    unlinked = []
    for track in eligible:
        if track in partner:
            continue
        top = best.get(track)
        if top is None or top["median_iou"] <= 0:
            reason = f"no {camera_b} box overlapped it on a compared frame (missed by the {camera_b} detector?)"
        elif (camera_a, track, camera_b) in rejected:
            reason = "lost a one-to-one conflict"
        else:
            reason = (f"best {camera_b} pair {display(camera_b, top['track'])}: median IoU {top['median_iou']:.2f} "
                      f"over {top['shared_frames']} compared frames (below a threshold)")
        unlinked.append({"track": display(camera_a, track), "frames_inside_overlap": inside_count[track],
                         "best_pair": top, "reason": reason})
    n_linked = sum(1 for t in eligible if t in partner)
    return {"camera": camera_a, "other_camera": camera_b, "overlap_region": [round(v, 1) for v in region],
            "uncut_boxes_inside_overlap": sum(inside_count.values()),
            "eligible_tracks": len(eligible), "linked": n_linked,
            "link_rate": round(n_linked / len(eligible), 4) if eligible else None, "unlinked_eligible": unlinked}


def read_events(run_dir: Path) -> list[dict]:
    """The events of <run_dir>/events.json that name a track (entity_id, else person_id); [] when missing."""
    path = Path(run_dir) / "events.json"
    if not path.exists():
        return []
    events = []
    for event in json.loads(path.read_text(encoding="utf-8")).get("events") or []:
        entity = event.get("entity_id") if event.get("entity_id") is not None else event.get("person_id")
        try:
            track = int(entity)
        except (TypeError, ValueError):
            continue
        events.append({"event_id": event.get("event_id"), "behavior": event.get("behavior"), "track_id": track,
                       "start_s": event.get("start_s"), "end_s": event.get("end_s"),
                       "confidence": event.get("confidence")})
    return events


def match_in_window(obs_a: list[dict], track_a: int, start_s: float, end_s: float, obs_b: list[dict],
                    region: Iterable[float], camera_b: str, min_iou: float = DEFAULTS["min_iou"],
                    min_frames: int = DEFAULTS["event_min_frames"], edge_margin_px: float = DEFAULTS["edge_margin_px"],
                    pad_s: float = 0.0) -> dict:
    """Which camera_b track has the same box as track_a of camera_a during [start_s, end_s] (frame by frame)?

    status: "same_swimmer" (a camera_b track: >= min_frames compared frames, median IoU >= min_iou, most agreeing
    frames wins), "not_in_view" (the track's uncut boxes never lie inside the overlap with camera_b),
    "too_few_frames" (inside the overlap on fewer than min_frames frames) or "no_match" (in view, no box coincides:
    camera_b's detector missed the swimmer, or the box differs).
    """
    region = list(region) if region is not None else None
    mine = {o["frame"]: o for o in obs_a
            if o["track_id"] == track_a and start_s - pad_s <= o["t"] <= end_s + pad_s}
    in_view = sum(1 for o in mine.values() if region and not o["cut"] and inside(o["box"], region, edge_margin_px))
    result = {"camera_id": camera_b, "frames_in_window": len(mine), "frames_inside_overlap": in_view,
              "status": None, "track_id": None, "display": None, "compared_frames": None, "median_iou": None,
              "agree_frames": None}
    if in_view == 0:
        result["status"] = "not_in_view"
        return result
    per_track: dict[int, list[float]] = {}
    for ob in obs_b:
        oa = mine.get(ob["frame"])
        if oa is None or oa["cut"] or ob["cut"]:
            continue
        per_track.setdefault(ob["track_id"], []).append(box_iou(oa["box"], ob["box"]))
    best = None
    for track_b, ious in per_track.items():
        median = statistics.median(ious)
        agree = sum(1 for v in ious if v >= min_iou)
        if len(ious) >= int(min_frames) and median >= float(min_iou):
            key = (agree, median, -track_b)
            if best is None or key > best[0]:
                best = (key, track_b, len(ious), median, agree)
    if best is None:
        result["status"] = "too_few_frames" if in_view < int(min_frames) else "no_match"
        return result
    _, track_b, n, median, agree = best
    result.update(status="same_swimmer", track_id=track_b, display=display(camera_b, track_b), compared_frames=n,
                  median_iou=round(median, 4), agree_frames=agree)
    return result


def event_links(camera_ids: list[str], runs_root: Path, observations: Mapping[str, list[dict]],
                views: Mapping[str, Mapping[str, Iterable[float]]], member_of: Mapping[tuple[str, int], str],
                status_of: Mapping[tuple[str, int], str], fps: Mapping[str, float],
                min_iou: float = DEFAULTS["min_iou"], min_frames: int = DEFAULTS["event_min_frames"],
                edge_margin_px: float = DEFAULTS["edge_margin_px"]) -> tuple[list[dict], list[dict]]:
    """(every events.json event with its swimmer in the other views, groups of events that are one swimmer).

    `views[camera]` maps every other compared camera to the crop overlap region. For each event, the event's track
    is compared frame by frame with the other camera's boxes INSIDE the event window (match_in_window), so a
    track that switched swimmers at another moment does not matter. Two events of different cameras are the same
    swimmer when the track of one coincides with the track of the other during either event window.
    """
    events = []
    for camera_id in camera_ids:
        pad = 0.5 / fps[camera_id] if fps.get(camera_id) else 0.0
        for event in read_events(runs_root / camera_id):
            key = (camera_id, event["track_id"])
            entry = {"camera_id": camera_id, **event, "display": display(*key),
                     "track_global_id": member_of.get(key), "track_identity_status": status_of.get(key, NO_LINK_STATUS),
                     "other_views": []}
            if event["start_s"] is not None and event["end_s"] is not None:
                for other, region in sorted((views.get(camera_id) or {}).items()):
                    entry["other_views"].append(match_in_window(
                        observations[camera_id], event["track_id"], float(event["start_s"]), float(event["end_s"]),
                        observations[other], region, other, min_iou, min_frames, edge_margin_px, pad))
            events.append(entry)

    parent = list(range(len(events)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    reasons: dict[tuple[int, int], str] = {}
    for i, first in enumerate(events):
        for j, second in enumerate(events):
            if i == j or first["camera_id"] == second["camera_id"]:
                continue
            for view in first["other_views"]:
                if view["status"] == "same_swimmer" and view["camera_id"] == second["camera_id"] \
                        and view["track_id"] == second["track_id"]:
                    parent[find(j)] = find(i)
                    reasons[(min(i, j), max(i, j))] = (
                        f"during {first['camera_id']} event {first['event_id']} ({first['start_s']}-{first['end_s']} s) "
                        f"{first['display']} and {view['display']} have the same box: median IoU "
                        f"{view['median_iou']:.2f} over {view['compared_frames']} frames")
    groups: dict[int, list[int]] = {}
    for i in range(len(events)):
        groups.setdefault(find(i), []).append(i)
    same = []
    for members in groups.values():
        if len({events[i]["camera_id"] for i in members}) < 2:
            continue
        windows = [(float(events[i]["start_s"]), float(events[i]["end_s"])) for i in members]
        same.append({"cameras": sorted({events[i]["camera_id"] for i in members}),
                     "events": [{k: events[i][k] for k in ("camera_id", "event_id", "behavior", "display",
                                                           "start_s", "end_s", "track_global_id")} for i in members],
                     "all_windows_overlap": max(s for s, _ in windows) <= min(e for _, e in windows),
                     "evidence": [reasons[k] for k in sorted(reasons) if k[0] in members and k[1] in members],
                     "note": "Same swimmer by known camera geometry during the event (not by appearance, not checked "
                             "by a person)."})
    same.sort(key=lambda g: (min(e["start_s"] for e in g["events"]), g["cameras"]))
    return events, same


# ---------------------------------------------------------------------------- main build

def build_links(camera_list: list[dict], geometry: Mapping[str, Any], runs_root: str | Path,
                min_shared_frames: int = DEFAULTS["min_shared_frames"], min_iou: float = DEFAULTS["min_iou"],
                edge_margin_px: float = DEFAULTS["edge_margin_px"], fps_tolerance: float = DEFAULTS["fps_tolerance"],
                n_examples: int = DEFAULTS["n_examples"], merge_gap_s: float = DEFAULTS["merge_gap_s"],
                event_min_frames: int = DEFAULTS["event_min_frames"], echo=print) -> dict:
    """Map every camera's boxes into source pixels, link tracks across overlapping cameras, assign global ids,
    and name the swimmer of every events.json event in the other views.

    A camera without geometry or without a readable run folder is reported and skipped.
    """
    runs_root = camreg.resolve_path(runs_root)
    geo_cams = geometry["cameras"]
    camera_rows, observations, metas, used = [], {}, {}, []
    for camera in camera_list:
        cam_id = str(camera["id"])
        row = {"camera_id": cam_id, "run_dir": _project(runs_root / cam_id), **camreg.sync_info(camera),
               "crop_xyxy": None, "proc_size": None, "n_tracks": 0, "n_boxes": 0, "n_boxes_cut": 0, "error": None,
               "warnings": []}
        camera_rows.append(row)
        if cam_id not in geo_cams:
            row["error"] = "no entry in the geometry file"
            echo(_ascii(f"  {cam_id}: SKIPPED ({row['error']})"))
            continue
        crop = geo_cams[cam_id]["crop_xyxy"]
        row["crop_xyxy"] = crop
        try:
            meta, rows = read_stage1(runs_root / cam_id)
        except (GeometryError, OSError, json.JSONDecodeError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            echo(_ascii(f"  {cam_id}: SKIPPED ({row['error']})"))
            continue
        row["proc_size"] = meta["proc_size"]
        proc_w, proc_h = (float(v) for v in meta["proc_size"])
        crop_w, crop_h = crop[2] - crop[0], crop[3] - crop[1]
        if abs(proc_w / proc_h - crop_w / crop_h) > 0.02 * (crop_w / crop_h):
            row["warnings"].append(f"proc_size {meta['proc_size']} has another aspect than the crop {crop_w}x{crop_h}")
        video, registry_video = str(meta.get("video") or ""), camera.get("video_path")
        if video and registry_video and Path(video).resolve() != Path(registry_video).resolve():
            row["warnings"].append(f"run video {video} is not the registry video {camera.get('video')}")
        obs = map_observations(cam_id, meta, rows, crop, edge_margin_px)
        observations[cam_id], metas[cam_id] = obs, meta
        used.append(camera)
        row.update(n_tracks=len({o["track_id"] for o in obs}), n_boxes=len(obs),
                   n_boxes_cut=sum(1 for o in obs if o["cut"]))
        echo(_ascii(f"  {cam_id}: {row['n_tracks']} tracks, {row['n_boxes']} boxes "
                    f"({row['n_boxes_cut']} cut by the crop edge), proc {meta['proc_size']} -> crop {crop}"))

    track_frames: dict[tuple[str, int], set] = {}
    track_first: dict[tuple[str, int], float] = {}
    track_last: dict[tuple[str, int], float] = {}
    for cam_id, obs in observations.items():
        for ob in obs:
            key = (cam_id, ob["track_id"])
            track_frames.setdefault(key, set()).add(ob["frame"])
            track_first[key] = min(track_first.get(key, ob["t"]), ob["t"])
            track_last[key] = max(track_last.get(key, ob["t"]), ob["t"])

    pairs, candidates, compared_all = [], [], []
    views: dict[str, dict[str, list[float]]] = {}            # camera -> {compared other camera: overlap region}
    for i, cam_a in enumerate(used):
        for cam_b in used[i + 1:]:
            a, b = str(cam_a["id"]), str(cam_b["id"])
            region = crop_overlap(geo_cams[a]["crop_xyxy"], geo_cams[b]["crop_xyxy"])
            pair = {"camera_a": a, "camera_b": b, "overlap_region": region, "compared": False, "reason": None,
                    "track_pairs_seen_together": 0, "candidate_links": 0, "links": 0}
            pairs.append(pair)
            if region is None:
                pair["reason"] = "crops do not overlap"
                continue
            ok, reason = alignment(cam_a, cam_b, metas[a], metas[b], fps_tolerance)
            pair["reason"] = reason
            if not ok:
                echo(_ascii(f"  {a}-{b}: not compared ({reason})"))
                continue
            stats = compare_tracks(observations[a], observations[b])
            links, compared = candidate_links(a, b, stats, min_shared_frames, min_iou, merge_gap_s)
            pair.update(compared=True, track_pairs_seen_together=len(stats), candidate_links=len(links))
            candidates.extend(links)
            compared_all.extend(compared)
            views.setdefault(a, {})[b] = region
            views.setdefault(b, {})[a] = region

    links, conflicts = resolve_one_to_one(candidates, track_frames)
    for link in candidates:
        link.pop("_agree_frames", None)                       # internal frame set, not written to JSON
    for pair in pairs:
        pair["links"] = sum(1 for l in links if l["camera_a"] == pair["camera_a"] and l["camera_b"] == pair["camera_b"])
    pair_stats = {_pair_key((p["camera_a"], p["track_a"]), (p["camera_b"], p["track_b"])): p["evidence"]
                  for p in compared_all}
    global_ids, member_of, component_conflicts = assign_global_ids(links, track_frames, track_first, pair_stats,
                                                                   min_shared_frames, min_iou)
    conflicts.extend(component_conflicts)
    in_conflict_component = {tuple(m) for c in component_conflicts for m in c["members"]}
    # A link inside a contradictory group (two tracks of ONE camera at the same moment would be 'the same swimmer')
    # is not a verified identity: it stays as evidence, marked unverified, and is not a geometry partner.
    for link in links:
        if (link["camera_a"], link["track_a"]) in in_conflict_component:
            link["status"] = identity.UNVERIFIED

    partners: dict[tuple[str, int], list[str]] = {}
    for link in links:
        if link["status"] == identity.UNVERIFIED:
            continue
        partners.setdefault((link["camera_a"], link["track_a"]), []).append(display(link["camera_b"], link["track_b"]))
        partners.setdefault((link["camera_b"], link["track_b"]), []).append(display(link["camera_a"], link["track_a"]))
    tracks, status_of = [], {}
    for key in sorted(track_frames):
        status = LINK_STATUS if key in member_of else (CONFLICT_STATUS if key in in_conflict_component
                                                       else NO_LINK_STATUS)
        status_of[key] = status
        tracks.append({"camera_id": key[0], "track_id": key[1], "display": display(*key),
                       "global_id": member_of.get(key), "status": status, "n_boxes": len(track_frames[key]),
                       "first_s": round(track_first[key], 3), "last_s": round(track_last[key], 3),
                       "geometry_partners": sorted(partners.get(key, []))})

    counts = {}
    for row in camera_rows:
        own = [t for t in tracks if t["camera_id"] == row["camera_id"]]
        counts[row["camera_id"]] = {"tracks": len(own),
                                    "linked": sum(1 for t in own if t["geometry_partners"]),
                                    "with_global_id": sum(1 for t in own if t["global_id"]),
                                    "unavailable": sum(1 for t in own if t["status"] == NO_LINK_STATUS),
                                    "conflict": sum(1 for t in own if t["status"] == CONFLICT_STATUS),
                                    "error": row["error"]}

    rejected = set()
    for conflict in conflicts:
        rej = conflict.get("rejected")
        if rej:
            rejected.add((rej["camera_a"], rej["track_a"], rej["camera_b"]))
            rejected.add((rej["camera_b"], rej["track_b"], rej["camera_a"]))
    link_rates = []
    for pair in pairs:
        if not pair["compared"]:
            continue
        a, b = pair["camera_a"], pair["camera_b"]
        for mine, other in ((a, b), (b, a)):
            link_rates.append(pair_link_rate(mine, other, pair["overlap_region"], observations[mine], links,
                                             compared_all, min_shared_frames, edge_margin_px, rejected))
        own_links = sorted((l for l in links if l["camera_a"] == a and l["camera_b"] == b),
                           key=lambda l: -l["evidence"]["agree_frames"])
        pair["examples"] = [{"link": f"{l['display_a']} <-> {l['display_b']}", "shared_frames":
                             l["evidence"]["shared_frames"], "agree_frames": l["evidence"]["agree_frames"],
                             "median_iou": l["evidence"]["median_iou"], "first_s": l["evidence"]["first_s"],
                             "last_s": l["evidence"]["last_s"]}
                            for l in own_links[: int(n_examples)]]

    camera_ids = [str(c["id"]) for c in used]
    fps = {cam: float(metas[cam].get("fps") or 0.0) for cam in camera_ids}
    events, same_swimmer = event_links(camera_ids, runs_root, observations, views, member_of, status_of, fps,
                                       min_iou, event_min_frames, edge_margin_px)
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": geometry.get("label") or "Camera geometry links",
        "method": METHOD,
        "method_text": ("Boxes of every camera mapped into shared source pixels with the known crop geometry; "
                        "tracks of two overlapping, synchronized cameras are linked when their boxes coincide "
                        "(median IoU) on enough shared frames."),
        "geometry_file": _project(geometry["path"]),
        "ground_truth": geometry.get("ground_truth") or "none",
        "thresholds": {"min_shared_frames": int(min_shared_frames), "min_iou": float(min_iou),
                       "edge_margin_px": float(edge_margin_px), "fps_tolerance": float(fps_tolerance),
                       "merge_gap_s": float(merge_gap_s), "event_min_frames": int(event_min_frames)},
        "ranking": ("one partner per (track, other camera): most agreeing frames (IoU >= min_iou) first, then the "
                    "higher median IoU (score)"),
        "cameras": camera_rows,
        "camera_pairs": pairs,
        "links": links,
        "global_ids": global_ids,
        "tracks": tracks,
        "per_camera_track_counts": counts,
        "conflicts": conflicts,
        "link_rates": link_rates,
        "events": events,
        "same_swimmer_events": same_swimmer,
        "caveats": list(CAVEATS),
    }


def _project(path: str | Path) -> str:
    """A path relative to the project root when inside it (forward slashes), else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(camreg.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------- Re-ID comparison

def compare_with_reid(links_doc: Mapping[str, Any], reid_doc: Mapping[str, Any],
                      top_n: int = DEFAULTS["reid_top_n"]) -> dict:
    """Agreement of reid.py's ranked appearance candidates with the geometry links (NOT accuracy).

    For every linked (camera, track) and each of its geometry partners: the partner's rank among the track's
    appearance candidates (all candidates, and cross-camera candidates only). reid keeps only its top_k candidates,
    so a partner missing from the list has rank > top_k (or was filtered).
    """
    ranked: dict[tuple[str, int], dict] = {}
    for track in reid_doc.get("tracks") or []:
        ranked[(str(track["camera_id"]), int(track["track_id"]))] = track
    queries = []
    for link in links_doc.get("links") or []:
        for (cam, trk), (p_cam, p_trk) in ((("camera_a", "track_a"), ("camera_b", "track_b")),
                                           (("camera_b", "track_b"), ("camera_a", "track_a"))):
            query, partner = (link[cam], int(link[trk])), (link[p_cam], int(link[p_trk]))
            entry = ranked.get(query)
            row = {"query": display(*query), "geometry_partner": display(*partner), "camera_pair":
                   f"{query[0]}->{partner[0]}", "geometry_median_iou": link["score"],
                   "rank_all": None, "rank_cross_camera": None, "partner_score": None, "top1": None,
                   "note": None}
            if entry is None:
                row["note"] = "track not in reid_candidates.json"
            elif not entry.get("candidates"):
                row["note"] = entry.get("note") or "no appearance candidates"
            else:
                cands = entry["candidates"]
                all_keys = [(str(c["camera_id"]), int(c["track_id"])) for c in cands]
                cross = [(str(c["camera_id"]), int(c["track_id"])) for c in cands
                         if str(c["camera_id"]) != query[0]]
                row["rank_all"] = all_keys.index(partner) + 1 if partner in all_keys else None
                row["rank_cross_camera"] = cross.index(partner) + 1 if partner in cross else None
                if partner in all_keys:
                    row["partner_score"] = cands[all_keys.index(partner)].get("score")
                row["top1"] = display(*all_keys[0])
                row["top1_score"] = cands[0].get("score")
            queries.append(row)

    def summarize(rows: list[dict]) -> dict:
        usable = [r for r in rows if r["note"] is None]
        n = len(usable)
        top1 = sum(1 for r in usable if r["rank_all"] == 1)
        topn = sum(1 for r in usable if r["rank_all"] is not None and r["rank_all"] <= top_n)
        x1 = sum(1 for r in usable if r["rank_cross_camera"] == 1)
        xn = sum(1 for r in usable if r["rank_cross_camera"] is not None and r["rank_cross_camera"] <= top_n)
        return {"n_queries": n, "skipped": len(rows) - n,
                "partner_is_top1": top1, f"partner_in_top{top_n}": topn,
                "top1_agreement": round(top1 / n, 4) if n else None,
                f"top{top_n}_agreement": round(topn / n, 4) if n else None,
                "partner_is_top1_cross_camera": x1, f"partner_in_top{top_n}_cross_camera": xn,
                "top1_agreement_cross_camera": round(x1 / n, 4) if n else None,
                f"top{top_n}_agreement_cross_camera": round(xn / n, 4) if n else None}

    by_pair = {}
    for pair in sorted({r["camera_pair"] for r in queries}):
        by_pair[pair] = summarize([r for r in queries if r["camera_pair"] == pair])
    settings = reid_doc.get("settings") or {}
    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "label": links_doc.get("label"),
        "title": "Agreement of appearance Re-ID candidates with geometry links (not accuracy)",
        "reid_method": reid_doc.get("method"),
        "reid_top_k": settings.get("top_k"),
        "geometry_method": links_doc.get("method"),
        "geometry_thresholds": links_doc.get("thresholds"),
        "top_n": int(top_n),
        "overall": summarize(queries),
        "by_camera_pair": by_pair,
        "queries": queries,
        "caveats": [
            "This is agreement with geometry links, NOT accuracy: geometry links are automatic (detector boxes + "
            "known crop geometry) and were not checked against human annotation; none exists.",
            "Every geometry link is asked in both directions (A->B and B->A), so each link counts twice.",
            f"reid keeps only its top_k ({settings.get('top_k')}) candidates per track; a partner outside them has "
            "rank null. 'rank_all' also counts same-camera candidates; 'rank_cross_camera' only other cameras.",
            "The views are crops of ONE recording (same lighting, same pixels), so appearance agreement here is far "
            "easier than at a real multi-camera site.",
        ],
    }


# ---------------------------------------------------------------------------- CLI

def summary_lines(doc: Mapping[str, Any]) -> list[str]:
    """Plain ASCII console summary of a build_links() result."""
    lines = [doc["label"], f"method {doc['method']}, thresholds {doc['thresholds']}"]
    for cam, count in doc["per_camera_track_counts"].items():
        state = f"ERROR {count['error']}" if count["error"] else \
            (f"{count['tracks']} tracks, {count['linked']} linked, {count['with_global_id']} with global id, "
             f"{count['conflict']} in conflict")
        lines.append(f"  {cam}: {state}")
    for pair in doc["camera_pairs"]:
        if pair["compared"]:
            lines.append(f"  {pair['camera_a']}-{pair['camera_b']}: {pair['links']} links "
                         f"({pair['candidate_links']} candidates, {pair['track_pairs_seen_together']} track pairs "
                         "seen together)")
        else:
            lines.append(f"  {pair['camera_a']}-{pair['camera_b']}: not compared ({pair['reason']})")
    for rate in doc["link_rates"]:
        value = "n/a" if rate["link_rate"] is None else f"{rate['link_rate']:.0%}"
        lines.append(f"  link rate {rate['camera']}->{rate['other_camera']}: {rate['linked']}/"
                     f"{rate['eligible_tracks']} eligible tracks ({value})")
    lines.append(f"  {len(doc['links'])} links, {len(doc['global_ids'])} global ids, {len(doc['conflicts'])} conflicts")
    for event in doc["events"]:
        seen = ", ".join(f"{v['camera_id']} {v['display']} (IoU {v['median_iou']:.2f}, {v['compared_frames']} fr)"
                         if v["status"] == "same_swimmer" else f"{v['camera_id']} {v['status']}"
                         for v in event["other_views"])
        lines.append(f"  event {event['camera_id']}#{event['event_id']} {event['behavior']} {event['display']} "
                     f"{event['start_s']}-{event['end_s']} s, track global id {event['track_global_id']}: {seen}")
    for group in doc["same_swimmer_events"]:
        text = "; ".join(f"{e['camera_id']} {e['behavior']} {e['display']} {e['start_s']}-{e['end_s']} s"
                         for e in group["events"])
        lines.append(f"  SAME SWIMMER in {len(group['cameras'])} cameras: {text}")
    return [_ascii(line) for line in lines]


def main(argv=None) -> int:
    """CLI: build geometry links (+ optional Re-ID agreement). Exit 0 ok, 1 some camera skipped, 2 bad input."""
    parser = argparse.ArgumentParser(description="AquaWatch: cross-camera links from known camera geometry")
    parser.add_argument("--cameras", required=True, metavar="cameras.yaml", help="camera registry")
    parser.add_argument("--geometry", required=True, metavar="JSON", help="crop geometry (crop_xyxy per camera)")
    parser.add_argument("--runs-root", required=True, metavar="DIR", help="run folders are <runs-root>/<camera id>")
    parser.add_argument("--out", required=True, metavar="JSON", help="where to write the geometry links")
    parser.add_argument("--min-shared-frames", type=int, default=DEFAULTS["min_shared_frames"],
                        help=f"compared frames needed for a link (default {DEFAULTS['min_shared_frames']})")
    parser.add_argument("--min-iou", type=float, default=DEFAULTS["min_iou"],
                        help=f"median IoU needed for a link (default {DEFAULTS['min_iou']})")
    parser.add_argument("--edge-margin", type=float, default=DEFAULTS["edge_margin_px"], dest="edge_margin_px",
                        help=f"source pixels from a crop border that count as cut (default {DEFAULTS['edge_margin_px']})")
    parser.add_argument("--fps-tolerance", type=float, default=DEFAULTS["fps_tolerance"],
                        help=f"relative fps difference allowed (default {DEFAULTS['fps_tolerance']})")
    parser.add_argument("--reid", metavar="JSON", help="reid_candidates.json to compare with "
                        "(default <runs-root>/reid_candidates.json when it exists)")
    parser.add_argument("--reid-out", metavar="JSON", help="Re-ID agreement output "
                        "(default reid_vs_geometry.json next to --out)")
    parser.add_argument("--db", metavar="PATH", help="also store the links (+ global ids, same-swimmer events) in this "
                                                    "SQLite event index (event_index.py)")
    parser.add_argument("--top-n", type=int, default=DEFAULTS["reid_top_n"],
                        help=f"top-N for the Re-ID agreement (default {DEFAULTS['reid_top_n']})")
    parser.add_argument("--examples", type=int, default=DEFAULTS["n_examples"], help="examples per camera pair")
    parser.add_argument("--merge-gap", type=float, default=DEFAULTS["merge_gap_s"], dest="merge_gap_s",
                        help=f"seconds joining agreement spans in the evidence (default {DEFAULTS['merge_gap_s']})")
    parser.add_argument("--event-min-frames", type=int, default=DEFAULTS["event_min_frames"],
                        help="compared frames inside an event window needed to name its swimmer in another view "
                             f"(default {DEFAULTS['event_min_frames']})")
    args = parser.parse_args(argv)

    try:
        registry = camreg.load_registry(args.cameras, quiet=True)
        geometry = load_geometry(camreg.resolve_path(args.geometry))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(_ascii(f"ERROR: {exc}"))
        return 2
    print(_ascii(f"Geometry links: {len(registry['cameras'])} camera(s) from {args.cameras}"))
    doc = build_links(registry["cameras"], geometry, args.runs_root, args.min_shared_frames, args.min_iou,
                      args.edge_margin_px, args.fps_tolerance, args.examples, args.merge_gap_s,
                      args.event_min_frames)
    out = camreg.resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    for line in summary_lines(doc):
        print(line)
    print(_ascii(f"wrote {out}"))
    if args.db:                                     # the query engine and the dashboard read the links from the index
        from event_index import EventIndex
        index = EventIndex(camreg.resolve_path(args.db))
        stored = index.store_links(doc["links"], replace=True, methods=[METHOD])   # no links found = none kept
        index.set_meta("geometry_global_ids", doc["global_ids"])
        index.set_meta("geometry_same_swimmer_events", doc["same_swimmer_events"])
        index.set_meta("geometry_caveats", doc["caveats"])
        print(_ascii(f"stored {stored} link(s) in {args.db}"))

    reid_path = camreg.resolve_path(args.reid) if args.reid else camreg.resolve_path(args.runs_root) / "reid_candidates.json"
    # The index holds the candidates of the LAST ingest --reid; a reid_candidates.json can be days older, so the
    # index is used when it has them (bug hunt 2026-10-11, multicam low).
    reid_doc, reid_source = None, None
    if args.db:
        rows = EventIndex(camreg.resolve_path(args.db)).identities()
        if any(r.get("candidates") for r in rows):
            reid_doc = {"tracks": [{"camera_id": r["camera_id"], "track_id": r["local_track_id"],
                                    "candidates": r["candidates"]} for r in rows]}
            reid_source = f"{args.db} (identities table)"
    if reid_doc is None and reid_path.exists():
        reid_doc, reid_source = json.loads(reid_path.read_text(encoding="utf-8")), _project(reid_path)
    if reid_doc is not None:
        comparison = compare_with_reid(doc, reid_doc, args.top_n)
        comparison["reid_file"] = reid_source
        reid_out = camreg.resolve_path(args.reid_out) if args.reid_out else out.parent / "reid_vs_geometry.json"
        reid_out.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        o = comparison["overall"]
        print(_ascii(f"Agreement with geometry links ({o['n_queries']} queries): partner is top-1 appearance "
                     f"candidate {o['partner_is_top1']}, in top-{args.top_n} {o[f'partner_in_top{args.top_n}']}"))
        print(_ascii(f"wrote {reid_out}"))
    else:
        print(_ascii(f"no Re-ID candidates at {reid_path}; agreement not computed"))
    return 1 if any(row["error"] for row in doc["cameras"]) else 0


if __name__ == "__main__":
    sys.exit(main())
