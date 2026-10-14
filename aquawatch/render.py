"""Evidence rendering: annotated video, per-event snapshots and plots, heatmap, timeline.

OpenCV does the drawing and the video I/O; Matplotlib (Agg backend, no GUI) draws the
charts. This file only presents what the earlier stages already decided: it never applies
a threshold of its own, it just draws the numbers stored in the events and tracks.

Entry point: render_all(video_path, tracker_result, tracks, final, zones, cfg, out_dir).

Reusable pieces (highlights.py uses them so the reel looks exactly like annotated.mp4):
    make_context(...)  -> DrawContext   everything the per-frame drawing needs, built once
    draw_frame(frame, frame_idx, t, ctx, k=1.0)   privacy blur first, then all overlays
    transcode_h264(src, dst)            mp4v -> H.264 with ffmpeg (browsers / Colab can play it)

All words shown to the user come from utils.entity_name / behavior_name / unit_name, so the
same code speaks "Person" and "Running" on a campus and "Animal" and "Stampede" in a pen.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # headless backend; must be selected before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

import distress  # noqa: E402
import privacy  # noqa: E402
import utils  # noqa: E402
from utils import (BEHAVIOR_COLORS_BGR, BEHAVIOR_COLORS_HEX, ensure_dir, fmt_time,  # noqa: E402
                   fmt_time_precise, id_color)

FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_BGR = BEHAVIOR_COLORS_BGR["zone_intrusion"]   # zones are always drawn in the intrusion colour
CROWD_BGR = BEHAVIOR_COLORS_BGR["crowding"]         # a crowded zone is filled in the crowding colour
LINK_NEAR_BGR = (0, 0, 255)                         # near-miss line when the two are close: red
LINK_FAR_BGR = (0, 190, 255)                        # ... fading to amber as they move apart
LINK_FADE = 2.5                                     # amber is reached at this many times the "near" distance
PRIORITY = ("near_miss", "fall", "zone_intrusion", "running", "loitering", "crowding")   # which colour wins
PRIVACY_PAD_FRAMES = 5   # processed frames region-blurred before a track starts and after it ends
PLOT_PAD_S = 3.0          # plots show the event +/- this many seconds
SNAPSHOT_MAX_W = 1280     # snapshots / heatmap use the original frame, capped to this width
JPEG_QUALITY = 90
MAX_SNAPSHOT_GAP_S = 1.5  # a track sample further than this from the snapshot time is not drawn
PLOT_RC = {"font.size": 8, "axes.titlesize": 9, "axes.grid": True, "grid.alpha": 0.25,
           "axes.spines.top": False, "axes.spines.right": False}


# =========================================================================== small helpers

def _warn(msg: str) -> None:
    """Print a non-fatal rendering problem (one bad artefact must not abort the run)."""
    print(f"[render] warning: {msg}")


def _imwrite(path: Path, img: np.ndarray) -> None:
    """Write a JPEG/PNG. imencode + tofile also works for non-ASCII Windows paths."""
    ext = Path(path).suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def _pt(p, k: float) -> tuple[int, int]:
    """Scale a processing-pixel point by k and round to the int pixel OpenCV expects."""
    return int(round(float(p[0]) * k)), int(round(float(p[1]) * k))


def _text_color_for(bg_bgr) -> tuple[int, int, int]:
    """Black text on light backgrounds, white on dark ones (keeps labels readable)."""
    b, g, r = bg_bgr
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 150 else (255, 255, 255)


def _label(img, text: str, x: float, y: float, bg, scale: float = 0.5, thickness: int = 1) -> int:
    """Draw a filled text label whose bottom-left corner is (x, y), kept inside the image.

    Returns the label's top y so the next label can be stacked above it.
    """
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    pad = max(2, int(round(4 * scale)))
    box_w, box_h = tw + 2 * pad, th + base + 2 * pad
    h_img, w_img = img.shape[:2]
    x0 = int(min(max(0, x), max(0, w_img - box_w)))
    y1 = int(min(max(box_h, y), h_img - 1))
    y0 = y1 - box_h
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y1), bg, -1)
    cv2.putText(img, text, (x0 + pad, y1 - pad - base), FONT, scale, _text_color_for(bg), thickness, cv2.LINE_AA)
    return y0


def _fit_scale(text: str, scale: float, thickness: int, max_w: float) -> float:
    """Shrink a font scale until `text` fits in max_w pixels (never below 0.3)."""
    while scale > 0.3 and cv2.getTextSize(text, FONT, scale, thickness)[0][0] > max_w:
        scale *= 0.92
    return scale


def _draw_zones(img, zones, k: float = 1.0, emphasise: str | None = None, crowded=()) -> None:
    """Semi-transparent zone polygons with an outline and the zone name (in place).

    `crowded` = names of zones that are over-crowded right now: they get a stronger fill in the
    crowding colour and a thick outline, so the reader sees which area the count refers to.
    """
    if not zones:
        return
    overlay = img.copy()
    polys = []
    for z in zones:
        pts = np.array([_pt(p, k) for p in z["points"]], dtype=np.int32).reshape(-1, 1, 2)
        polys.append(pts)
        cv2.fillPoly(overlay, [pts], ZONE_BGR)
    alpha = 0.22
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    crowd_polys = [pts for z, pts in zip(zones, polys) if z.get("name") in crowded]
    if crowd_polys:
        overlay = img.copy()
        cv2.fillPoly(overlay, crowd_polys, CROWD_BGR)
        img[:] = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    for z, pts in zip(zones, polys):
        is_crowd = z.get("name") in crowded
        strong = is_crowd or (emphasise is not None and z.get("name") == emphasise)
        colour = CROWD_BGR if is_crowd else ZONE_BGR
        cv2.polylines(img, [pts], True, colour, max(2, int(round((4 if is_crowd else 3 if strong else 2) * k))),
                      cv2.LINE_AA)
        x0, y0 = pts.reshape(-1, 2).min(axis=0)
        _label(img, f"ZONE: {z.get('name', 'zone')}", int(x0), int(y0), colour, scale=0.45 * max(k, 1.0))


def _draw_trail(img, pts: np.ndarray, color, k: float, fade: bool) -> None:
    """Draw a polyline of foot points; with fade=True it thickens towards the newest point."""
    if len(pts) < 2:
        return
    pts = np.round(pts * k).astype(np.int32)
    if not fade:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, max(2, int(round(2 * k))), cv2.LINE_AA)
        return
    n = len(pts) - 1
    for j in range(n):
        thick = max(1, int(round((1 + 2.0 * (j + 1) / n) * k)))
        cv2.line(img, tuple(pts[j]), tuple(pts[j + 1]), color, thick, cv2.LINE_AA)


UNDER_WATER_BGR = (60, 20, 255)      # Rule C "possibly under water": urgent red, not the Level 1 amber
UNDER_WATER_LABEL = "POSSIBLY UNDER WATER"


def _under_water(ev) -> bool:
    """Is this a Rule C alert (seen again at the same spot): its own urgent kind (distress.alert_kind)."""
    return ev.get("behavior") == "aquatic_warning" and distress.alert_kind(ev) == "under_water"


def _event_color(ev) -> tuple[int, int, int]:
    if _under_water(ev):
        return UNDER_WATER_BGR
    return BEHAVIOR_COLORS_BGR.get(ev.get("behavior"), (0, 200, 0))


def _ev_label(cfg, ev) -> str:
    """The overlay label of one event: the behaviour name, or POSSIBLY UNDER WATER for a Rule C alert."""
    return UNDER_WATER_LABEL if _under_water(ev) else _beh_label(cfg, ev.get("behavior"))


def _draw_spot(img, ev, k: float, s: float, label: bool = True) -> None:
    """Red double ring (+ label) where a Rule C swimmer vanished (metrics.spot, processing px)."""
    spot = (ev.get("metrics") or {}).get("spot")
    if not spot:
        return
    centre = _pt(spot, k)
    radius = max(14, int(round(0.035 * img.shape[1])))
    cv2.circle(img, centre, radius, (255, 255, 255), max(3, int(round(4 * s))), cv2.LINE_AA)
    cv2.circle(img, centre, radius, UNDER_WATER_BGR, max(2, int(round(2 * s))), cv2.LINE_AA)
    cv2.circle(img, centre, max(3, radius // 5), UNDER_WATER_BGR, -1, cv2.LINE_AA)
    if not label:
        return
    _label(img, f"{UNDER_WATER_LABEL} (E{ev['event_id']})", centre[0] - radius, centre[1] - radius, UNDER_WATER_BGR,
           scale=0.5 * s)


def _draw_spot_marks(img, t_now: float, ctx, k: float, s: float) -> None:
    """Mark the spot of every Rule C alert from its alert time for pool.mark_spot_s seconds."""
    keep = float(((ctx.cfg.get("pool") or {}).get("mark_spot_s")) or 0.0)
    for ev in ctx.events:
        if not _under_water(ev):
            continue
        alert_t = float((ev.get("metrics") or {}).get("alert_time_s", ev["end_s"]))
        if alert_t - ctx.tol <= t_now <= alert_t + keep:
            _draw_spot(img, ev, k, s)


def _priority(ev) -> int:
    """Sort key: the most important behaviour decides the box colour."""
    beh = ev.get("behavior")
    return PRIORITY.index(beh) if beh in PRIORITY else len(PRIORITY)


def _beh_label(cfg, key: str) -> str:
    """Upper-case behaviour name for overlays, in the scenario's own wording."""
    return utils.behavior_name(cfg, key).upper()


def _cls_of(track) -> int:
    """Class id of a track (0 = person when the track has no class information)."""
    return int(getattr(track, "cls", 0) or 0)


def _aspect(track) -> np.ndarray:
    """Width / height of the track's boxes: the smoothed Track.aspect, or the raw boxes if it is missing."""
    asp = getattr(track, "aspect", None)
    if asp is not None and len(asp) == len(track.t):
        return np.asarray(asp, dtype=float)
    w = track.box[:, 2] - track.box[:, 0]
    h = np.maximum(track.box[:, 3] - track.box[:, 1], 1e-6)
    return w / h


def _track_name(cfg, tracks, tid) -> str:
    """"Person #3" / "Car #7" for a track id."""
    tr = tracks.get(tid)
    return utils.entity_name(cfg, tid, _cls_of(tr) if tr is not None else None)


def _event_entities(ev) -> list[int]:
    """Track ids an event is about (both objects of a near miss, every member of a crowd)."""
    ents = [i for i in (ev.get("entities") or [ev.get("entity_id")]) if i is not None]
    other = ev.get("other_entity")
    if other is not None and other not in ents:
        ents.append(other)
    return [int(i) for i in ents]


def _who(ev, cfg, tracks) -> str:
    """Display name of who an event is about: "Person #3", "Person #3 & Car #7", "Group of 6"."""
    if ev.get("behavior") == "crowding" or ev.get("entity_id") is None:
        return ev.get("entity_name") or f"Group of {len(ev.get('entities') or [])}"
    name = ev.get("entity_name") or _track_name(cfg, tracks, ev["entity_id"])
    other = ev.get("other_entity")
    if ev.get("behavior") == "near_miss" and other is not None:
        other_name = _track_name(cfg, tracks, other)
        if other_name not in name:
            name = f"{name} & {other_name}"
    return name


def _is_vulnerable(track, cfg) -> bool:
    """True for the class(es) a near miss protects (people, by default)."""
    ids = ((cfg.get("behaviors") or {}).get("near_miss") or {}).get("vulnerable_classes", [0])
    return _cls_of(track) in [int(i) for i in ids]


def _pair_scale(tra, ia, trb, ib, cfg) -> float:
    """Size unit for a near-miss distance: A's scale, or the mean of both when both are people."""
    if _is_vulnerable(tra, cfg) and _is_vulnerable(trb, cfg):
        return 0.5 * (float(tra.height[ia]) + float(trb.height[ib]))
    return float(tra.height[ia])


def _link_color(d: float, near: float) -> tuple[int, int, int]:
    """Near-miss line colour: red when d <= near, fading to amber as the pair moves apart."""
    if not np.isfinite(d) or near <= 0:
        return LINK_FAR_BGR
    u = min(1.0, max(0.0, (d - near) / ((LINK_FADE - 1.0) * near)))
    return tuple(int(round(a + (b - a) * u)) for a, b in zip(LINK_NEAR_BGR, LINK_FAR_BGR))


def _nearest_sample(track, t: float, max_gap_s: float = MAX_SNAPSHOT_GAP_S) -> int | None:
    """Index of the track sample closest in time to t, or None if the track has nothing near t."""
    if track is None or len(track.t) == 0:
        return None
    i = int(np.argmin(np.abs(track.t - t)))
    return i if abs(float(track.t[i]) - t) <= max_gap_s else None


def _index_tracks_by_frame(tracks) -> dict[int, list[tuple[int, int]]]:
    """frame_idx -> [(track_id, sample index)] so each video frame finds its boxes in O(1)."""
    lookup: dict[int, list[tuple[int, int]]] = {}
    for tid, tr in tracks.items():
        for i, f in enumerate(tr.frame):
            lookup.setdefault(int(f), []).append((tid, i))
    return lookup


def _detections_by_frame(tracker_result: dict) -> dict[int, list[tuple[int, tuple[float, ...], int]]]:
    """Raw tracked boxes by frame, including short tracks omitted from behaviour analysis."""
    lookup: dict[int, list[tuple[int, tuple[float, ...], int]]] = {}
    for row in (tracker_result or {}).get("detections") or []:
        if len(row) < 8:
            continue
        box = tuple(float(value) for value in row[3:7])
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        cls = int(row[8]) if len(row) >= 9 else 0
        lookup.setdefault(int(row[0]), []).append((int(row[2]), box, cls))
    return lookup


# =========================================================================== drawing context

@dataclass
class DrawContext:
    """Everything draw_frame needs, built once per run by make_context()."""
    cfg: dict
    tracks: dict
    zones: list
    events: list
    lookup: dict                 # frame_idx -> [(track_id, sample index)]
    zone_by_name: dict
    fps: float
    stride: int
    trail_s: float
    tol: float                   # an event counts as "active" within this many seconds of its ends
    privacy: bool
    detections: dict = field(default_factory=dict)       # frame_idx -> [(track_id, box, class_id)]
    person_boxes: dict = field(default_factory=dict)   # frame_idx -> [(x1, y1, x2, y2)] processing px (privacy only)
    frame_times: dict = field(default_factory=dict)    # frame_idx -> time in s (for the crowd-count plot)


def _grow_box(box, factor: float) -> tuple:
    """Box scaled around its centre (privacy margin for extrapolated boxes)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 * factor, (y2 - y1) / 2 * factor
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _region(near) -> tuple:
    """Union of a few boxes, enlarged 30%, tagged "body" (blur all of it): where a person probably is
    in a frame that has no box for them."""
    union = (min(b[0] for b in near), min(b[1] for b in near), max(b[2] for b in near), max(b[3] for b in near))
    return _grow_box(union, 1.3) + ("body",)


def _person_boxes_by_frame(tracker_result: dict, tracks: dict, person_cls=frozenset({0}),
                           head_cls=frozenset()) -> dict:
    """Every person box per frame, for privacy blurring.

    The raw detections are used (not only the kept tracks) so that short ghost tracks and
    brief passers-by are blurred too. Only people have a face to hide: COCO class 0, or every
    body-state class of a custom swimmer detector (`person_cls`, see make_context).
    """
    out: dict[int, list] = {}
    rows = (tracker_result or {}).get("detections") or []
    if len(rows):
        stride = max(1, int((tracker_result or {}).get("stride") or 1))
        by_id: dict[int, list] = {}
        for row in rows:
            if len(row) >= 9 and int(row[8]) not in person_cls:
                continue
            box = (row[3], row[4], row[5], row[6])
            # a head_only box IS the head: a 5-tuple is blurred whole (apply_privacy), not only its top 28 %
            out.setdefault(int(row[0]), []).append(box + (1,) if len(row) >= 9 and int(row[8]) in head_cls else box)
            by_id.setdefault(int(row[2]), []).append((int(row[0]), box))
        # ByteTrack only reports a person once the track is confirmed, so the first sighting (and the
        # frames around a dropout) have no box. Copy each track's edge boxes a few processed frames
        # backwards / forwards so the face is already covered when the person first appears.
        for items in by_id.values():
            items.sort(key=lambda fb: fb[0])
            frames = [f for f, _ in items]
            boxes = [np.asarray(b, dtype=float) for _, b in items]
            last = len(items) - 1
            for n, f in enumerate(frames):
                # Only frames where this person has NO box get a region blur (when tracked, the head blur
                # is enough): before the first sighting, after the last one, and inside a dropout.
                if n == 0:
                    region = _region(boxes[0:3])
                    for k in range(1, PRIVACY_PAD_FRAMES + 1):
                        out.setdefault(f - k * stride, []).append(region)
                if n == last:
                    region = _region(boxes[max(0, last - 2):last + 1])
                    for k in range(1, PRIVACY_PAD_FRAMES + 1):
                        out.setdefault(f + k * stride, []).append(region)
                elif frames[n + 1] - f > stride:
                    region = _region(boxes[n:n + 2])
                    for g in range(f + stride, frames[n + 1], stride):
                        out.setdefault(g, []).append(region)
        return out
    for tr in tracks.values():
        if _cls_of(tr) in person_cls:
            for f, box in zip(tr.frame, tr.box):
                out.setdefault(int(f), []).append(tuple(box))
    return out


def make_context(tracker_result: dict, tracks: dict, final: dict, zones: list, cfg: dict) -> DrawContext:
    """Collect tracks, events, zones and privacy boxes into one object for draw_frame()."""
    tracker_result = tracker_result or {}
    zones = zones or []
    stride = max(1, int(tracker_result.get("stride") or 1))
    fps = float(tracker_result.get("fps") or 0) or 30.0
    use_privacy = privacy.is_enabled(cfg)
    # with a swimmer state detector every class is a person (horizontal, vertical, head only, ...)
    person_cls = frozenset(utils.state_classes(cfg)) or frozenset({0})
    head_cls = frozenset(c for c, name in utils.state_classes(cfg).items() if name == "head_only")
    lookup = _index_tracks_by_frame(tracks)
    frame_times = {int(f): float(t) for tr in tracks.values() for f, t in zip(tr.frame, tr.t)}
    return DrawContext(
        cfg=cfg, tracks=tracks, zones=zones, events=(final or {}).get("events", []) or [],
        lookup=lookup, zone_by_name={z.get("name"): z for z in zones},
        fps=fps, stride=stride, trail_s=float((cfg.get("output") or {}).get("trail_s", 3.0)),
        tol=0.5 * stride / fps, privacy=use_privacy,
        detections=_detections_by_frame(tracker_result),
        person_boxes=_person_boxes_by_frame(tracker_result, tracks, person_cls, head_cls) if use_privacy else {},
        frame_times=frame_times)


def apply_privacy(img, frame_idx: int, k: float, ctx: DrawContext) -> None:
    """Blur the heads in `img` (a frame at k x processing size) when privacy mode is on. In place."""
    if not ctx.privacy:
        return
    entries = ctx.person_boxes.get(frame_idx, [])
    heads = [(b[0] * k, b[1] * k, b[2] * k, b[3] * k) for b in entries if len(b) == 4]
    regions = [(b[0] * k, b[1] * k, b[2] * k, b[3] * k) for b in entries if len(b) > 4]
    privacy.anonymize(img, heads, ctx.cfg)
    if regions:                                   # unconfirmed sightings: blur the whole region
        body_cfg = dict(ctx.cfg, privacy=dict(ctx.cfg["privacy"], mode="body"))
        privacy.anonymize(img, regions, body_cfg)


def _zone_members(ctx: DrawContext, present: dict, zone_name) -> list[int]:
    """Track ids (among `present` = {track_id: sample index}) whose foot is inside the named zone.

    A name that is not a drawn zone (the "whole frame" case) counts everybody in the frame.
    """
    zone = ctx.zone_by_name.get(zone_name)
    if zone is None:
        return list(present)
    return [tid for tid, i in present.items()
            if utils.point_in_polygon(float(ctx.tracks[tid].foot[i][0]), float(ctx.tracks[tid].foot[i][1]),
                                      zone["points"])]


# =========================================================================== frame drawing

def _draw_near_link(img, ev, tra, ia, trb, ib, cfg, k: float) -> float:
    """Line between the two foot points, red when near, with the distance (and POSSIBLE CONTACT). Returns d."""
    s = max(k, 1.0)
    ncfg = (cfg.get("behaviors") or {}).get("near_miss") or {}
    near = float(ncfg.get("near_distance_bh", 0.6))
    scale = _pair_scale(tra, ia, trb, ib, cfg)
    pa, pb = _pt(tra.foot[ia], k), _pt(trb.foot[ib], k)
    d = float(np.hypot(*(np.asarray(tra.foot[ia]) - np.asarray(trb.foot[ib])))) / scale if scale > 0 else float("nan")
    col = _link_color(d, near)
    cv2.line(img, pa, pb, col, max(2, int(round(3 * k))), cv2.LINE_AA)
    for p in (pa, pb):
        cv2.circle(img, p, max(3, int(round(4 * k))), col, -1, cv2.LINE_AA)
        cv2.circle(img, p, max(3, int(round(4 * k))) + 1, (255, 255, 255), 1, cv2.LINE_AA)
    if np.isfinite(d):
        text = f"{d:.2f} {utils.unit_name(cfg)}"
        if d <= float(ncfg.get("contact_bh", 0.15)):
            text += "  POSSIBLE CONTACT"
        _label(img, text, (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0, col, scale=0.5 * s,
               thickness=max(1, int(round(s))))
    return d


def _draw_crowd_label(img, ev, count: int, ctx: DrawContext, k: float) -> None:
    """"CROWDING: 6 in 'zone'" label inside the zone (or under the clock for a whole-frame count)."""
    s = max(k, 1.0)
    zname = ev.get("zone")
    text = f"{_beh_label(ctx.cfg, 'crowding')}: {count} in '{zname}'"
    zone = ctx.zone_by_name.get(zname)
    if zone is not None:
        x0, y0 = np.array([_pt(p, k) for p in zone["points"]]).min(axis=0)
        _label(img, text, int(x0), int(y0) + int(28 * s), CROWD_BGR, scale=0.55 * s, thickness=max(1, int(round(s))))
    else:                                    # no polygon to point at: frame the whole picture
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), CROWD_BGR, max(3, int(round(4 * s))))
        _label(img, text, int(10 * s), int(70 * s), CROWD_BGR, scale=0.55 * s, thickness=max(1, int(round(s))))


def _box_name(ctx: DrawContext, tid: int) -> str:
    """Short box label: "#3"; with several classes in the scene ("Car #7") so the kind is visible.
    A swimmer state detector has one kind of entity, so its boxes stay "#3" like the stock pool preset."""
    classes = ((ctx.cfg.get("model") or {}).get("classes")) or [0]
    several_kinds = len(classes) > 1 and not utils.state_classes(ctx.cfg)
    return _track_name(ctx.cfg, ctx.tracks, tid) if several_kinds else f"#{tid}"


def draw_frame(frame, frame_idx: int, t_now: float, ctx: DrawContext, k: float = 1.0):
    """Annotate one processed frame exactly like annotated.mp4 does. In place; returns `frame`.

    `frame` is the video frame at k x the processing size (k = 1 for the video and the highlight
    reel). Order: privacy blur FIRST (so no overlay is blurred and no face is left), then zones,
    trails, boxes, labels, near-miss lines, crowd counts, and the clock.
    """
    apply_privacy(frame, frame_idx, k, ctx)
    s = max(k, 1.0)
    cfg = ctx.cfg
    active = [e for e in ctx.events if e["start_s"] - ctx.tol <= t_now <= e["end_s"] + ctx.tol]
    present = dict(ctx.lookup.get(frame_idx, []))            # track_id -> sample index in this frame

    crowds = []                                              # (event, members) for every active crowding event
    crowd_member: set[int] = set()
    for e in active:
        if e.get("behavior") == "crowding":
            members = _zone_members(ctx, present, e.get("zone"))
            crowds.append((e, len(members)))
            crowd_member.update(members)
    _draw_zones(frame, ctx.zones, k, crowded={e.get("zone") for e, _ in crowds})

    by_track: dict[int, list] = {}
    for e in active:
        if e.get("behavior") != "crowding":
            for tid in _event_entities(e):
                by_track.setdefault(tid, []).append(e)

    for tid, i in present.items():
        tr = ctx.tracks[tid]
        evs = sorted(by_track.get(tid, []), key=_priority)
        id_col = id_color(tid)
        col = _event_color(evs[0]) if evs else CROWD_BGR if tid in crowd_member else id_col

        # trail = smoothed foot points of the last trail_s seconds, within the same segment
        sel = (tr.t >= t_now - ctx.trail_s) & (tr.t <= t_now) & (tr.segment == tr.segment[i])
        _draw_trail(frame, tr.foot[sel], col, k, fade=True)
        cv2.circle(frame, _pt(tr.foot[i], k), max(3, int(round(3 * k))), col, -1, cv2.LINE_AA)

        x1, y1, x2, y2 = (int(round(v * k)) for v in tr.box[i])
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(2, int(round((3 if evs or tid in crowd_member else 2) * k))),
                      cv2.LINE_AA)
        top = _label(frame, _box_name(ctx, tid), x1, y1, id_col, scale=0.5 * s)
        for e in evs:  # active events stack above the ID label
            top = _label(frame, f"{_ev_label(cfg, e)} (E{e['event_id']})", x1, top, _event_color(e), scale=0.5 * s)

    classes = ((cfg.get("model") or {}).get("classes")) or [0]
    several_kinds = len(classes) > 1 and not utils.state_classes(cfg)
    for tid, box, cls in ctx.detections.get(frame_idx, []):
        if tid in present:
            continue
        x1, y1, x2, y2 = (int(round(value * k)) for value in box)
        col = id_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, max(2, int(round(2 * k))), cv2.LINE_AA)
        name = utils.entity_name(cfg, tid, cls) if several_kinds else f"#{tid}"
        _label(frame, name, x1, y1, col, scale=0.5 * s)

    for e in active:                                         # near-miss lines (both objects must be in view)
        if e.get("behavior") == "near_miss" and e.get("entity_id") in present and e.get("other_entity") in present:
            a, b = e["entity_id"], e["other_entity"]
            _draw_near_link(frame, e, ctx.tracks[a], present[a], ctx.tracks[b], present[b], cfg, k)
    for e, count in crowds:
        _draw_crowd_label(frame, e, count, ctx, k)
    _draw_spot_marks(frame, t_now, ctx, k, s)                  # Rule C: where the swimmer vanished

    # heads-up display: timestamp and number of active events
    text = f"{fmt_time_precise(t_now)}   events active: {len(active)}"
    (tw, th), base = cv2.getTextSize(text, FONT, 0.55 * s, 1)
    x_end, y_end = min(frame.shape[1], tw + int(16 * s)), min(frame.shape[0], th + base + int(12 * s))
    frame[:y_end, :x_end] = (frame[:y_end, :x_end] * 0.35).astype(np.uint8)
    cv2.putText(frame, text, (int(8 * s), th + int(6 * s)), FONT, 0.55 * s, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _caption_bar(img, text: str, col, s: float) -> None:
    """Dark bar across the top with a colour tab on the left (used for snapshots)."""
    thick = max(1, int(round(s)))
    scale = _fit_scale(text, 0.55 * s, thick, img.shape[1] - 28 * s)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    bar_h = th + base + int(14 * s)
    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (30, 30, 30), -1)
    cv2.rectangle(img, (0, 0), (int(10 * s), bar_h), col, -1)  # behaviour-colour tab on the left
    cv2.putText(img, text, (int(18 * s), th + int(7 * s)), FONT, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _make_snapshot(img, k, ev, ctx: DrawContext, frame_idx) -> np.ndarray:
    """Evidence image for one event. `img` is the frame at scale k x processing size (already privacy-blurred)."""
    cfg, tracks = ctx.cfg, ctx.tracks
    s = max(k, 1.0)
    thick = max(1, int(round(s)))
    beh = ev.get("behavior")
    col = _event_color(ev)
    label = _ev_label(cfg, ev)
    snap_t = ev.get("snapshot_time_s")
    snap_t = float(snap_t) if snap_t is not None else 0.5 * (ev["start_s"] + ev["end_s"])
    present = dict(ctx.lookup.get(frame_idx, []))
    focus = _event_entities(ev)
    focus_set = set(focus)

    _draw_zones(img, ctx.zones, k, emphasise=ev.get("zone"), crowded={ev.get("zone")} if beh == "crowding" else ())

    # context: everybody else in the frame, thin boxes
    for tid, i in present.items():
        if tid in focus_set:
            continue
        x1, y1, x2, y2 = tracks[tid].box[i] * k
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), id_color(tid), max(1, int(round(k))), cv2.LINE_AA)
        _label(img, _box_name(ctx, tid), int(x1), int(y1), id_color(tid), scale=0.45 * s)

    # the entities of the event: full trail during the event, thick box in the behaviour colour
    drawn: dict[int, int] = {}
    for n, tid in enumerate(focus):
        tr = tracks.get(tid)
        i = _nearest_sample(tr, snap_t)
        if i is None:
            continue
        drawn[tid] = i
        if beh != "crowding":
            span = (tr.t >= ev["start_s"]) & (tr.t <= ev["end_s"])
            _draw_trail(img, tr.foot[span], col if beh != "near_miss" else id_color(tid), k, fade=False)
            if span.any():  # white ring on the first point so the walking direction is clear
                cv2.circle(img, _pt(tr.foot[span][0], k), max(4, int(5 * k)), (255, 255, 255), 2, cv2.LINE_AA)
        x1, y1, x2, y2 = (int(round(v * k)) for v in tr.box[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(3, int(round(4 * k))), cv2.LINE_AA)
        cv2.circle(img, _pt(tr.foot[i], k), max(4, int(5 * k)), col, -1, cv2.LINE_AA)
        if beh == "crowding":
            continue                                       # one count label for the whole group, below
        name = _box_name(ctx, tid)
        _label(img, f"{name} {label}" if n == 0 else name, x1, y1, col, scale=0.55 * s, thickness=thick)

    if beh == "near_miss":
        a, b = ev.get("entity_id"), ev.get("other_entity")
        if a in drawn and b in drawn:
            _draw_near_link(img, ev, tracks[a], drawn[a], tracks[b], drawn[b], cfg, k)
    elif beh == "crowding":
        in_zone = _zone_members(ctx, present, ev.get("zone"))
        _draw_crowd_label(img, ev, len(in_zone) if in_zone else len(drawn), ctx, k)
    if _under_water(ev):
        _draw_spot(img, ev, k, s, label=False)            # the swimmer's own box already carries the label

    start = ev.get("start") or fmt_time(ev["start_s"])
    end = ev.get("end") or fmt_time(ev["end_s"])
    caption = (f"Event {ev['event_id']} | {_who(ev, cfg, tracks)} {label} | {start}-{end}"
               f" | conf {float(ev.get('confidence', 0.0)):.2f}")
    if ev.get("severity"):
        caption += f" | {str(ev['severity']).upper()}"
    _caption_bar(img, caption, col, s)
    return img


# =========================================================================== video + snapshot pass

BROWSER_CODECS = {"h264", "avc1", "x264", "vp80", "vp90", "vp09", "av01"}   # FourCCs that Chrome / Edge / Firefox play


def video_codec(path) -> str:
    """FourCC of a video file in lower case ("h264", "fmp4" = OpenCV's mp4v, ...); "" when it cannot be opened."""
    cap = cv2.VideoCapture(str(path))
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.isOpened() else 0
    finally:
        cap.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()


def opencv_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 with OpenCV alone, for machines without ffmpeg. True when dst is a playable H.264 file.

    On Windows this uses the built-in Media Foundation encoder (asked for directly, so OpenCV does not first try a
    missing OpenH264 library and print errors); elsewhere OpenCV picks whatever H.264 encoder it has.
    """
    src, dst = Path(src), Path(dst)
    cap = cv2.VideoCapture(str(src))
    writer = None
    written = 0
    try:
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # H.264 (yuv420p) needs an even width and height
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
        if w <= 0 or h <= 0:
            return False
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        if os.name == "nt":
            writer = cv2.VideoWriter(str(dst), cv2.CAP_MSMF, fourcc, fps, (w, h))
        else:
            writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
        if not writer.isOpened():
            return False
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(np.ascontiguousarray(frame[:h, :w]))
            written += 1
    except cv2.error as exc:
        _warn(f"OpenCV could not write H.264 ({exc})")
        return False
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return written > 0 and dst.is_file() and dst.stat().st_size > 0 and video_codec(dst) in BROWSER_CODECS


def transcode_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so browsers and Colab can play it: with ffmpeg when it is installed, else with OpenCV's own
    H.264 writer (see opencv_h264). True on success; False keeps the caller's mp4v file."""
    exe = shutil.which("ffmpeg")
    if exe and _ffmpeg_h264(exe, src, dst):
        return True
    return opencv_h264(src, dst)


def _ffmpeg_h264(exe: str, src: Path, dst: Path) -> bool:
    """Re-encode with ffmpeg to H.264/yuv420p (faststart, so playback starts before the whole file is loaded)."""
    cmd = [exe, "-y", "-loglevel", "error", "-i", str(src), "-an",
           "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # yuv420p needs even width and height
           "-movflags", "+faststart", str(dst)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        _warn(f"ffmpeg could not be run ({exc}); trying OpenCV H.264")
        return False
    if done.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        _warn(f"ffmpeg failed ({done.stderr.strip()[:200]}); trying OpenCV H.264")
        return False
    return True


_transcode_h264 = transcode_h264   # old private name, kept so nothing that imported it breaks


def _scaled_frame(frame, pw: int, ph: int) -> tuple[np.ndarray, float]:
    """Original frame resized for snapshots/heatmap (up to SNAPSHOT_MAX_W). Returns (img, k)."""
    sw = max(pw, min(frame.shape[1], SNAPSHOT_MAX_W))
    if sw == pw:
        return cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA), 1.0
    sh = int(round(frame.shape[0] * sw / frame.shape[1]))
    return cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA), sw / float(pw)


def _video_pass(video_path, tr_res, ctx: DrawContext, out: Path, want_video: bool, want_first_frame: bool) -> dict:
    """One sequential pass over the video.

    Writes annotated.mp4 (if want_video), the snapshots of all events, and keeps the first
    processed frame for the heatmap. Frames are read exactly as the tracker read them:
    every `stride`-th frame, resized to proc_size. Skipped frames are only grab()-ed.
    Privacy blur (when on) is applied to the frame before anything is drawn on it.
    """
    result = {"first_frame": None, "k": 1.0, "video": False, "snapshots": 0}
    events, cfg = ctx.events, ctx.cfg
    stride, fps = ctx.stride, ctx.fps
    limit = int(tr_res.get("n_frames_read") or 0)          # 0 = unknown, read to the end
    proc = tr_res.get("proc_size") or [int(cfg["video"]["resize_width"]), 0]
    pw, ph = int(proc[0]), int(proc[1])

    # which processed frame is nearest each event's snapshot time?
    last_frame = ((limit - 1) // stride) * stride if limit > 0 else None
    snap_jobs: dict[int, list] = {}
    for ev in events:
        t_snap = ev.get("snapshot_time_s")
        if t_snap is None:
            t_snap = 0.5 * (ev["start_s"] + ev["end_s"])
        f = max(0, int(round(float(t_snap) * fps / stride)) * stride)
        if last_frame is not None:
            f = min(f, last_frame)
        snap_jobs.setdefault(f, []).append(ev)
    pending = set(snap_jobs)

    if not (want_video or pending or want_first_frame):
        return result
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _warn(f"cannot open video {video_path}; skipping annotated video, snapshots and heatmap background")
        return result

    writer = None
    tmp_path = out / "annotated_tmp.mp4"
    if pending:
        ensure_dir(out / "snapshots")

    frame_idx = 0
    try:
        while limit <= 0 or frame_idx < limit:
            if frame_idx % stride != 0:
                if not cap.grab():
                    break
                frame_idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if ph <= 0:  # proc height unknown: derive it from the frame and the resize width
                ph = int(round(frame.shape[0] * pw / frame.shape[1]))
            t_now = frame_idx / fps

            if want_first_frame and result["first_frame"] is None:
                first, result["k"] = _scaled_frame(frame, pw, ph)
                apply_privacy(first, frame_idx, result["k"], ctx)
                result["first_frame"] = first

            if frame_idx in pending:                      # evidence snapshot(s) for this frame
                scaled, k = _scaled_frame(frame, pw, ph)
                apply_privacy(scaled, frame_idx, k, ctx)  # blur once, then every event draws on a copy
                for ev in snap_jobs[frame_idx]:
                    try:
                        snap = _make_snapshot(scaled.copy(), k, ev, ctx, frame_idx)
                        rel = f"snapshots/e{ev['event_id']}.jpg"
                        _imwrite(out / rel, snap)
                        ev["snapshot"] = rel
                        result["snapshots"] += 1
                    except Exception as exc:  # noqa: BLE001 - never lose the whole run for one image
                        _warn(f"snapshot for event {ev.get('event_id')} failed: {type(exc).__name__}: {exc}")
                pending.discard(frame_idx)

            if want_video:
                if writer is None:
                    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / stride, (pw, ph))
                    if not writer.isOpened():
                        _warn("cv2.VideoWriter could not open an mp4v file; no annotated video")
                        writer, want_video = None, False
                if writer is not None:
                    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
                    writer.write(draw_frame(small, frame_idx, t_now, ctx))

            frame_idx += 1
            if not want_video and not pending and (result["first_frame"] is not None or not want_first_frame):
                break  # nothing left to produce, do not decode the rest of the video
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if tmp_path.exists():
        final_path = out / "annotated.mp4"
        if tmp_path.stat().st_size == 0:
            tmp_path.unlink()
        elif transcode_h264(tmp_path, final_path):
            tmp_path.unlink()
            result["video"] = True
        else:  # no ffmpeg (or it failed): keep the OpenCV mp4v file
            os.replace(tmp_path, final_path)
            result["video"] = True
    return result


# =========================================================================== matplotlib charts

def _time_axis(ax, lo: float, hi: float) -> None:
    """Integer-second ticks formatted mm:ss."""
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: fmt_time(v)))


def _fit_ylim(ax, series, lines) -> None:
    """Y range that contains the data and every rule line, with some air; the axis starts at 0 if it can."""
    values = [float(np.nanmax(y)) for y in series if np.isfinite(y).any()] + [ln[0] for ln in lines]
    lows = [float(np.nanmin(y)) for y in series if np.isfinite(y).any()] + [ln[0] for ln in lines] + [0.0]
    top, bottom = max(values), min(lows)
    pad = 0.12 * max(top - bottom, 0.5)
    ax.set_ylim(bottom - (pad if bottom < 0 else 0), top + pad)


def _draw_rule_lines(ax, lines) -> None:
    for value, label, style, lcol in lines:
        ax.axhline(value, color=lcol, linestyle=style, linewidth=1.2, label=label)


def _no_data_note(ax) -> None:
    ax.text(0.5, 0.5, "no measurable data in this window", transform=ax.transAxes,
            ha="center", va="center", color="#777777")


def _plot_event(ev, ctx: DrawContext, path: Path) -> bool:
    """Evidence plot for one event: the measured quantity vs time, the rule line(s), the event span.

    Returns False (after a warning) when the event has no track data to plot.
    """
    beh = ev.get("behavior")
    cfg, tracks = ctx.cfg, ctx.tracks
    title = f"E{ev['event_id']}  {_who(ev, cfg, tracks)}  {utils.behavior_name(cfg, beh)}"
    if beh == "crowding":
        _plot_crowding(ev, ctx, title, path)
        return True
    if beh in ("aquatic_warning", "aquatic_distress", "submersion"):
        _plot_risk(ev, ctx, title, path)
        return True
    if beh == "approaching":
        track = tracks.get(ev.get("entity_id"))
        if track is None or len(track.t) == 0:
            _warn(f"event {ev.get('event_id')}: no track data; no plot")
            return False
        _plot_approaching(ev, track, ctx, title, path)
        return True
    if beh == "near_miss":
        tra, trb = tracks.get(ev.get("entity_id")), tracks.get(ev.get("other_entity"))
        if tra is None or trb is None or len(tra.t) == 0 or len(trb.t) == 0:
            _warn(f"event {ev.get('event_id')}: a track of the pair is missing; no plot")
            return False
        _plot_near_miss(ev, tra, trb, ctx, title, path)
        return True
    track = tracks.get(ev.get("entity_id"))
    if track is None or len(track.t) == 0:
        _warn(f"event {ev.get('event_id')}: {_track_name(cfg, tracks, ev.get('entity_id'))} has no track data; no plot")
        return False
    _plot_single(ev, track, ctx, title, path)
    return True


def _plot_single(ev, track, ctx: DrawContext, title: str, path: Path) -> None:
    """Loitering / zone intrusion / running / fall: one measured series against its rule line(s)."""
    beh = ev.get("behavior")
    cfg, zones = ctx.cfg, ctx.zones
    start, end = float(ev["start_s"]), float(ev["end_s"])
    lo, hi = max(0.0, start - PLOT_PAD_S), end + PLOT_PAD_S
    color = BEHAVIOR_COLORS_HEX.get(beh, "#1f77b4")
    metrics = ev.get("metrics") or {}
    bcfg = cfg["behaviors"]
    unit = utils.unit_name(cfg)

    sel = (track.t >= lo) & (track.t <= hi)
    t = track.t[sel]
    lines: list[tuple[float, str, str, str]] = []   # (value, label, linestyle, colour)

    zone = None
    if beh == "zone_intrusion":
        zone = next((z for z in zones if z.get("name") == ev.get("zone")), zones[0] if zones else None)

    if beh == "loitering":
        in_ev = (track.t >= start) & (track.t <= end)
        in_ev = in_ev if in_ev.any() else sel
        centre = track.foot[in_ev].mean(axis=0)
        bh = float(np.median(track.height[in_ev])) or 1.0
        y = np.hypot(*(track.foot[sel] - centre).T) / bh
        ylabel = f"distance from the spot ({unit})"
        radius = metrics.get("radius_threshold_bh", bcfg["loitering"]["radius_bh"])
        lines.append((float(radius), f"radius rule ({radius:g})", "--", "#c0392b"))
    elif beh == "zone_intrusion" and zone is not None:
        poly = zone["points"]
        # signed depth: positive = inside the zone, negative = outside (distance to the nearest edge / size)
        y = np.array([(1.0 if utils.point_in_polygon(float(p[0]), float(p[1]), poly) else -1.0)
                      * utils.distance_to_polygon_edge(float(p[0]), float(p[1]), poly) / max(float(h), 1e-6)
                      for p, h in zip(track.foot[sel], track.height[sel])])
        ylabel = f"depth inside the zone ({unit})"
        lines.append((0.0, "zone edge", "-", "#c0392b"))
    elif beh == "fall":
        y = _aspect(track)[sel]
        ylabel = "box width / height"
        fcfg = bcfg["fall"]
        down, up = float(fcfg["down_min_aspect"]), float(fcfg["upright_max_aspect"])
        lines.append((down, f"lying: w/h >= {down:g}", "--", "#c0392b"))
        lines.append((up, f"upright: w/h <= {up:g}", ":", "#2a7f3f"))
    else:  # running (and the fallback for anything else): speed in body sizes per second
        y = track.speed[sel]
        ylabel = f"speed ({unit} / s)"
        if beh == "running":
            start_thr = metrics.get("start_threshold_bh_s", bcfg["running"]["start_speed_bh_s"])
            end_thr = metrics.get("end_threshold_bh_s", bcfg["running"]["end_speed_bh_s"])
            lines.append((float(start_thr), f"running starts ({start_thr:g})", "--", "#c0392b"))
            lines.append((float(end_thr), f"running ends ({end_thr:g})", ":", "#e08e0b"))

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=110)
        try:
            ax.axvspan(start, end, color=color, alpha=0.16, label=f"event ({end - start:.1f} s)")
            _draw_rule_lines(ax, lines)
            if np.isfinite(y).any():
                ax.plot(t, y, color=color, linewidth=1.8, marker="o", markersize=2.2)
                _fit_ylim(ax, [y], lines)
            else:
                _no_data_note(ax)
            _time_axis(ax, lo, hi)
            ax.set_xlabel("video time (mm:ss)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, loc="left")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)


def _crowd_series(ctx: DrawContext, zone_name, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """(times, how many tracks have their foot in the zone) at every processed frame between lo and hi."""
    times, counts = [], []
    for f in sorted(ctx.frame_times, key=lambda f: ctx.frame_times[f]):
        t = ctx.frame_times[f]
        if lo <= t <= hi:
            times.append(t)
            counts.append(len(_zone_members(ctx, dict(ctx.lookup.get(f, [])), zone_name)))
    return np.array(times, dtype=float), np.array(counts, dtype=float)


def _plot_risk(ev, ctx: DrawContext, title: str, path: Path) -> None:
    """Pool events: the swimmer's distress risk over time against the watch / warning / alert lines."""
    pcfg = ctx.cfg.get("pool") or {}
    series = (ev.get("metrics") or {}).get("risk_series") or []
    start, end = float(ev["start_s"]), float(ev["end_s"])
    alert_t = float((ev.get("metrics") or {}).get("alert_time_s") or start)
    color = BEHAVIOR_COLORS_HEX.get(ev.get("behavior"), "#d00000")
    lines = [(float(pcfg.get("alert_risk", 0.7)), "alert level", "--", "#c0392b"),
             (float(pcfg.get("warning_risk", 0.55)), "warning", ":", "#e67e22"),
             (float(pcfg.get("watch_risk", 0.3)), "watch", ":", "#7f8c8d")]
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=110)
        try:
            ax.axvspan(start, max(end, start + 0.3), color=color, alpha=0.15,
                       label="warning signs" if ev.get("behavior") == "aquatic_warning" else "distress run")
            _draw_rule_lines(ax, lines)
            if series:
                t = [p[0] for p in series]
                r = [p[1] for p in series]
                ax.plot(t, r, color=color, linewidth=1.8, marker="o", markersize=2.2)
                ax.axvline(alert_t, color="#c0392b", linewidth=1.2, label=f"alert {utils.fmt_time_precise(alert_t)}")
                _time_axis(ax, min(t), max(max(t), end))
            else:
                _no_data_note(ax)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("video time (mm:ss)")
            ax.set_ylabel("distress risk")
            ax.set_title(title, loc="left")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)


def _plot_approaching(ev, track, ctx: DrawContext, title: str, path: Path) -> None:
    """Approaching (wearable camera): time to contact over time, against the alert threshold."""
    import approaching
    cfg = ctx.cfg
    start, end = float(ev["start_s"]), float(ev["end_s"])
    lo, hi = max(0.0, start - PLOT_PAD_S), end + PLOT_PAD_S
    sig = approaching.looming_signals(track, cfg, track.frame_size or (640, 360))
    keep = (sig["t"] >= lo) & (sig["t"] <= hi)
    t, ttc = sig["t"][keep], np.minimum(sig["ttc"][keep], 6.0)      # "not approaching" drawn at the top
    limit = float(cfg["behaviors"]["approaching"]["ttc_s"])
    color = BEHAVIOR_COLORS_HEX.get("approaching", "#ffc800")
    lines = [(limit, f"alert at <= {limit:g} s to contact", "--", "#c0392b")]
    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=110)
        try:
            ax.axvspan(start, end, color=color, alpha=0.18, label=f"alert ({end - start:.1f} s)")
            _draw_rule_lines(ax, lines)
            if len(t):
                ax.plot(t, ttc, color="#b8860b", linewidth=1.8, marker="o", markersize=2.2)
                ax.set_ylim(0, 6.2)
            else:
                _no_data_note(ax)
            _time_axis(ax, lo, hi)
            ax.set_xlabel("video time (mm:ss)")
            ax.set_ylabel("time to contact (s)")
            ax.set_title(title, loc="left")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)


def _plot_crowding(ev, ctx: DrawContext, title: str, path: Path) -> None:
    """Crowding: how many are inside the zone over time, against the min_count rule."""
    cfg = ctx.cfg
    start, end = float(ev["start_s"]), float(ev["end_s"])
    lo, hi = max(0.0, start - PLOT_PAD_S), end + PLOT_PAD_S
    metrics = ev.get("metrics") or {}
    min_count = float(metrics.get("min_count", cfg["behaviors"]["crowding"]["min_count"]))
    t, count = _crowd_series(ctx, ev.get("zone"), lo, hi)
    color = BEHAVIOR_COLORS_HEX["crowding"]
    lines = [(min_count, f"crowding rule (>= {min_count:g})", "--", "#c0392b")]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=110)
        try:
            ax.axvspan(start, end, color=color, alpha=0.16, label=f"event ({end - start:.1f} s)")
            _draw_rule_lines(ax, lines)
            if len(t):
                ax.plot(t, count, color=color, linewidth=1.8, drawstyle="steps-mid", marker="o", markersize=2.2)
                _fit_ylim(ax, [count], lines)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                _no_data_note(ax)
            _time_axis(ax, lo, hi)
            ax.set_xlabel("video time (mm:ss)")
            ax.set_ylabel(f"count inside '{ev.get('zone')}'")
            ax.set_title(title, loc="left")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)


def _pair_signals(tra, trb, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For the frames where both tracks exist: time, distance d(t) in body sizes, closing speed vc(t).

    Same definitions as the near-miss rule (DESIGN 4.8): distance between the smoothed foot points
    divided by A's scale (the mean of both when both are people); closing speed = how fast d fell
    over the last features.speed_window_s (positive while approaching, NaN when there is too little history).
    """
    common, ia, ib = np.intersect1d(tra.frame, trb.frame, return_indices=True)
    if len(common) == 0:
        empty = np.empty(0)
        return empty, empty, empty
    t = tra.t[ia]
    both = _is_vulnerable(tra, cfg) and _is_vulnerable(trb, cfg)
    scale = 0.5 * (tra.height[ia] + trb.height[ib]) if both else tra.height[ia]
    d = np.hypot(*(tra.foot[ia] - trb.foot[ib]).T) / np.maximum(scale, 1e-6)
    w = float(cfg["features"]["speed_window_s"])
    first = np.searchsorted(t, t - w - 1e-9, side="left")     # earliest sample inside the window
    dt = t - t[first]
    vc = np.full(len(t), np.nan)
    ok = (dt >= 0.5 * w - 1e-9) & (dt > 0)
    vc[ok] = -(d[ok] - d[first][ok]) / dt[ok]
    return t, d, vc


def _plot_near_miss(ev, tra, trb, ctx: DrawContext, title: str, path: Path) -> None:
    """Near miss: distance d(t) with the "near" line (top) and closing speed (bottom), event +/- 3 s."""
    cfg = ctx.cfg
    unit = utils.unit_name(cfg)
    ncfg = cfg["behaviors"]["near_miss"]
    start, end = float(ev["start_s"]), float(ev["end_s"])
    lo, hi = max(0.0, start - PLOT_PAD_S), end + PLOT_PAD_S
    metrics = ev.get("metrics") or {}
    color = BEHAVIOR_COLORS_HEX["near_miss"]

    t, d, vc = _pair_signals(tra, trb, cfg)
    win = (t >= lo) & (t <= hi)
    near = float(ncfg["near_distance_bh"])
    d_lines = [(near, f"near ({near:g})", "--", "#c0392b")]
    contact = float(ncfg.get("contact_bh", 0))
    if contact > 0:
        d_lines.append((contact, f"contact ({contact:g})", ":", "#7b1fa2"))
    vc_lines = [(float(ncfg["min_closing_speed_bh_s"]), f"min closing speed ({ncfg['min_closing_speed_bh_s']:g})",
                 "--", "#e08e0b"), (0.0, "", "-", "#999999")]
    peak = metrics.get("peak_time_s")

    with plt.rc_context(PLOT_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 3.9), dpi=110, sharex=True,
                                       gridspec_kw={"height_ratios": [3, 2]})
        try:
            for ax in (ax1, ax2):
                ax.axvspan(start, end, color=color, alpha=0.16, label=f"event ({end - start:.1f} s)" if ax is ax1 else None)
                if peak is not None:
                    ax.axvline(float(peak), color=color, linewidth=1.0, linestyle="-.",
                               label="closest moment" if ax is ax1 else None)
            _draw_rule_lines(ax1, d_lines)
            _draw_rule_lines(ax2, vc_lines)
            if win.any() and np.isfinite(d[win]).any():
                ax1.plot(t[win], d[win], color=color, linewidth=1.8, marker="o", markersize=2.2)
                _fit_ylim(ax1, [d[win]], d_lines)
            else:
                _no_data_note(ax1)
            if win.any() and np.isfinite(vc[win]).any():
                ax2.plot(t[win], vc[win], color="#555555", linewidth=1.6, marker="o", markersize=2.0)
                _fit_ylim(ax2, [vc[win]], vc_lines)
            ax1.set_ylabel(f"distance ({unit})")
            ax2.set_ylabel(f"closing speed\n({unit} / s)")
            ax2.set_xlabel("video time (mm:ss)")
            _time_axis(ax2, lo, hi)
            ax1.set_title(title, loc="left")
            ax1.legend(loc="upper right", fontsize=7, framealpha=0.9)
            ax2.legend(loc="upper right", fontsize=7, framealpha=0.9)
            fig.tight_layout()
            fig.savefig(path)
        finally:
            plt.close(fig)


def _render_heatmap(tracks, zones, base_img, k, pw, ph, path: Path) -> None:
    """Foot-point density blended over the first frame (brighter = more time spent there)."""
    if base_img is None:  # video could not be read: dark canvas instead of the first frame
        base_img, k = np.full((ph, pw, 3), 70, dtype=np.uint8), 1.0
    sh, sw = base_img.shape[:2]

    grid = np.zeros((ph, pw), dtype=np.float32)       # one cell per processing pixel
    n_points = 0
    for tr in tracks.values():
        if len(tr.foot) == 0:
            continue
        xi = np.clip(np.round(tr.foot[:, 0]).astype(int), 0, pw - 1)
        yi = np.clip(np.round(tr.foot[:, 1]).astype(int), 0, ph - 1)
        np.add.at(grid, (yi, xi), 1.0)                # every processed frame counts once = time spent
        n_points += len(xi)

    out = (base_img.astype(np.float32) * 0.7).astype(np.uint8)   # slightly dimmed so the colours stand out
    if n_points:
        grid = cv2.GaussianBlur(grid, (0, 0), sigmaX=max(4.0, 0.02 * pw))
        grid = grid / max(float(grid.max()), 1e-9)
        grid = grid ** 0.4                            # compress so brief walk-throughs stay visible next to a long stand
        heat = cv2.resize(grid, (sw, sh), interpolation=cv2.INTER_LINEAR)
        cmap = getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET)
        colour = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cmap).astype(np.float32)
        alpha = (np.clip(heat * 1.3, 0, 1) * 0.75 * (heat > 0.04))[..., None]
        out = (out.astype(np.float32) * (1 - alpha) + colour * alpha).astype(np.uint8)
    _draw_zones(out, zones, k)
    caption = "Foot-point density: brighter = more time spent" if n_points else "Nothing was tracked"
    _label(out, caption, 6, 6 + int(22 * max(k, 1.0)), (40, 40, 40), scale=0.5 * max(k, 1.0))
    _imwrite(path, out)


def _pack_rows(spans) -> tuple[list[int], int]:
    """Give every (start, end) span a sub-row so that overlapping spans never share one (greedy packing).

    Returns (sub-row index of each span, in the input order; number of sub-rows used, at least 1).
    """
    order = sorted(range(len(spans)), key=lambda i: (spans[i][0], spans[i][1]))
    row_end: list[float] = []
    rows = [0] * len(spans)
    for i in order:
        a, b = spans[i]
        for r, last in enumerate(row_end):
            if a >= last - 1e-9:
                rows[i], row_end[r] = r, b
                break
        else:
            rows[i] = len(row_end)
            row_end.append(b)
    return rows, max(len(row_end), 1)


MIN_BLOCK_S = 0.1          # events shorter than this are drawn this wide so they stay visible


def _render_timeline(tracks, events, duration_s: float, cfg: dict, path: Path) -> None:
    """One row per entity: thin grey bar while visible, coloured blocks for events.

    Events that overlap in time on the same row (loitering AND zone intrusion at once) are stacked
    as sub-rows inside that row, so neither hides the other. A near miss is drawn on both rows
    of the pair; a crowding event (no single entity) goes on a "Group" row at the top.
    """
    ids = set(tracks)
    for e in events:
        ids.update(i for i in _event_entities(e) if e.get("behavior") != "crowding")
    ids = sorted(ids, key=lambda i: (float(tracks[i].t[0]) if i in tracks and len(tracks[i].t) else 1e9, i))
    has_group = any(e.get("behavior") == "crowding" or e.get("entity_id") is None for e in events)
    rows = (["group"] if has_group else []) + ids               # first row is drawn at the top
    n = len(rows)

    def row_events(row):
        if row == "group":
            return [e for e in events if e.get("behavior") == "crowding" or e.get("entity_id") is None]
        return [e for e in events if e.get("behavior") != "crowding" and row in _event_entities(e)]

    ends = [float(tr.t[-1]) for tr in tracks.values() if len(tr.t)] + [float(e["end_s"]) for e in events]
    span = max(float(duration_s or 0.0), max(ends, default=0.0), 1.0)
    packed = {}
    for row in rows:
        evs = row_events(row)
        sub, m = _pack_rows([(float(e["start_s"]), max(float(e["end_s"]), float(e["start_s"]) + MIN_BLOCK_S)) for e in evs])
        packed[row] = (evs, sub, m)
    extra = sum(m - 1 for _evs, _sub, m in packed.values())      # sub-rows need a little more height
    row_h = max(0.12, min(0.38, 10.0 / max(n, 1)))
    fig_h = max(2.3, row_h * (n + 0.5 * extra) + 1.5)

    with plt.rc_context(PLOT_RC | {"axes.titlesize": 10}):
        fig, ax = plt.subplots(figsize=(10, fig_h), dpi=110)
        try:
            ypos = {row: n - 1 - r for r, row in enumerate(rows)}
            for row in rows:
                y = ypos[row]
                tr = tracks.get(row) if row != "group" else None
                if tr is not None and len(tr.t):
                    # one grey bar per continuous segment (a gap in tracking is not "visible")
                    cuts = np.flatnonzero(np.diff(tr.segment)) + 1
                    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(tr.t)]):
                        ax.broken_barh([(float(tr.t[a]), max(float(tr.t[b - 1] - tr.t[a]), 0.05))],
                                       (y - 0.12, 0.24), facecolors="#b4bcc6", linewidth=0)
                evs, sub, m = packed[row]
                lane = 0.74 / m                                   # height of one sub-row
                for e, r in zip(evs, sub):
                    c = BEHAVIOR_COLORS_HEX.get(e["behavior"], "#1f77b4")
                    width = max(float(e["end_s"] - e["start_s"]), MIN_BLOCK_S)
                    y0 = y - 0.37 + r * lane
                    ax.broken_barh([(float(e["start_s"]), width)], (y0 + 0.02, lane - 0.04), facecolors=c,
                                   edgecolors="#333333", linewidth=0.6)
                    if width > 0.04 * span:
                        ax.text(e["start_s"] + width / 2, y0 + lane / 2, f"E{e['event_id']}", ha="center",
                                va="center", fontsize=7 if m == 1 else 6, fontweight="bold",
                                color="white" if e["behavior"] != "loitering" else "black")
            ax.set_ylim(-0.7, max(n - 1, 0) + 0.7)
            if n <= 40:
                ax.set_yticks([ypos[r] for r in rows])
                ax.set_yticklabels([utils.entity_name(cfg, None) if r == "group" else _track_name(cfg, tracks, r)
                                    for r in rows])
            else:
                ax.set_yticks([])
            if n == 0:
                ax.text(0.5, 0.5, "Nothing was tracked in this video", transform=ax.transAxes,
                        ha="center", va="center", color="#777777")
            _time_axis(ax, 0.0, span)
            ax.set_xlabel("video time (mm:ss)")
            ax.set_title("Timeline: who was visible (grey) and when an incident happened (colour)", loc="left")
            handles = [Patch(facecolor="#b4bcc6", label="visible")]
            handles += [Patch(facecolor=BEHAVIOR_COLORS_HEX[b], edgecolor="#333333", label=utils.behavior_name(cfg, b))
                        for b in utils.BEHAVIORS]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False)
            fig.tight_layout()
            fig.savefig(path, bbox_inches="tight")
        finally:
            plt.close(fig)


# =========================================================================== public entry point

def render_all(video_path, tracker_result: dict, tracks: dict, final: dict, zones: list,
               cfg: dict, out_dir) -> dict:
    """Write all visual evidence into out_dir and fill snapshot / speed_plot on every event.

    Files: annotated.mp4, snapshots/e{id}.jpg, plots/e{id}.png, heatmap.jpg, timeline.png.
    With privacy mode on (cfg privacy.enabled), faces are blurred before anything is drawn.
    Safe with 0 tracks / 0 events, a missing ffmpeg, or an unreadable video: whatever cannot
    be produced is skipped with a warning and the rest is still written. Returns `final`.
    """
    out = ensure_dir(out_dir)
    events = final.get("events", []) if final else []
    zones = zones or []
    out_cfg = cfg.get("output", {})
    proc = tracker_result.get("proc_size") or [int(cfg["video"]["resize_width"]), 360]
    pw, ph = int(proc[0]), int(proc[1])
    ctx = make_context(tracker_result, tracks, final, zones, cfg)

    # 1) evidence plots (pure Matplotlib, needs no video)
    for ev in events:
        try:
            ensure_dir(out / "plots")
            rel = f"plots/e{ev['event_id']}.png"
            if _plot_event(ev, ctx, out / rel):
                ev["speed_plot"] = rel
        except Exception as exc:  # noqa: BLE001
            _warn(f"plot for event {ev.get('event_id')} failed: {type(exc).__name__}: {exc}")

    # 2) one pass over the video: annotated.mp4 + snapshots + first frame for the heatmap
    first_frame, k = None, 1.0
    try:
        res = _video_pass(video_path, tracker_result, ctx, out,
                          want_video=bool(out_cfg.get("annotated_video", True)),
                          want_first_frame=bool(out_cfg.get("heatmap", True)))
        first_frame, k = res["first_frame"], res["k"]
    except Exception as exc:  # noqa: BLE001
        _warn(f"video pass failed: {type(exc).__name__}: {exc}")

    # 3) heatmap and timeline
    if out_cfg.get("heatmap", True):
        try:
            _render_heatmap(tracks, zones, first_frame, k, pw, ph, out / "heatmap.jpg")
        except Exception as exc:  # noqa: BLE001
            _warn(f"heatmap failed: {type(exc).__name__}: {exc}")
    if out_cfg.get("timeline", True):
        try:
            _render_timeline(tracks, events, float(tracker_result.get("duration_s") or 0.0), cfg,
                             out / "timeline.png")
        except Exception as exc:  # noqa: BLE001
            _warn(f"timeline failed: {type(exc).__name__}: {exc}")
    return final
