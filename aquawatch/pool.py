"""Pool pipeline orchestrator: Aquatic Distress Behaviour Intelligence.

    stage 1  tracking   (tracker.py: YOLO + ByteTrack)          -> stage1_tracking.jsonl
    stage 2  behaviour  (pool_behaviour.py, pose keypoints here) -> stage2_behaviour.jsonl
    stage 3  reasoning  (distress.py: risk + state machine)      -> stage3_events.json
    stage 4  dashboard  (pool_dashboard.json + report/UI/announcement)

run_pool() is called by run.py when the scenario has pool.enabled. It returns the alerts as normal
standard events (behaviours "aquatic_distress" and "submersion"), so the evidence cards, snapshots,
highlight reel, report and web UI all work unchanged. The stage files follow the team's data contract,
so each teammate's module can also be run (and replaced) on its own.
"""
from __future__ import annotations

from pathlib import Path

import distress
import pool_behaviour
import stitch
import utils


# --------------------------------------------------------------------------- pose keypoints (optional)

def extract_keypoints(video_path, tracker_result: dict, tracks: dict, cfg: dict) -> dict:
    """Pose keypoints for every swimmer at pool.pose_hz: {track_id: {frame_idx: 17x3 list}} in processing px.

    Runs yolo11n-pose on a padded crop around each tracked box. Frames are read sequentially (one pass).
    Imports cv2 / ultralytics lazily. Raises on a missing model or video; run_pool turns that into a warning.
    """
    import cv2
    from ultralytics import YOLO
    import cpu_limits
    import tracker as trk
    cpu_limits.apply()                  # keep torch / Ultralytics on the laptop's thread budget
    from pose_verify import padded_crop_box

    pcfg = cfg["pool"]
    fps = float(tracker_result["fps"]) or 30.0
    stride = int(tracker_result.get("stride") or 1)
    every = max(stride, int(round(fps / float(pcfg["pose_hz"]) / stride)) * stride)
    wanted: dict[int, list] = {}
    for tid, tr in tracks.items():
        for f, box in zip(tr.frame, tr.box):
            if int(f) % every == 0:
                wanted.setdefault(int(f), []).append((tid, box))
    if not wanted:
        return {}
    model = YOLO(trk.resolve_weights(cfg["model"]["pose_weights"]))
    device = trk.resolve_device(tracker_result.get("device") or "cpu")
    cap, _, _, _ = trk.open_video(video_path)
    proc = tracker_result["proc_size"]
    out: dict[int, dict] = {}
    last = max(wanted)
    f = 0
    try:
        while f <= last:
            if f not in wanted:
                if not cap.grab():
                    break
                f += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            img = trk.resize_frame(frame, proc)
            for tid, box in wanted[f]:
                x1, y1, x2, y2 = padded_crop_box(box, img.shape[1], img.shape[0], 0.3)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                res = model.predict(crop, verbose=False, device=device, conf=0.2)[0]
                kp = _best_keypoints(res, (box[0] - x1, box[1] - y1, box[2] - x1, box[3] - y1))
                if kp is not None:
                    out.setdefault(int(tid), {})[f] = [[round(x + x1, 1), round(y + y1, 1), round(c, 3)]
                                                       for x, y, c in kp]
            f += 1
    finally:
        cap.release()
    return out


def _best_keypoints(result, target):
    """17x3 keypoints of the detection whose centre is inside the tracked box (else the most confident)."""
    import numpy as np
    from tracker import to_numpy
    boxes, kps = getattr(result, "boxes", None), getattr(result, "keypoints", None)
    if boxes is None or kps is None or len(boxes) == 0 or kps.xy is None or kps.conf is None:
        return None
    conf = to_numpy(boxes.conf).reshape(-1)
    xyxy = to_numpy(boxes.xyxy).reshape(-1, 4)
    xy, kc = to_numpy(kps.xy), to_numpy(kps.conf)
    cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    inside = (cx >= target[0]) & (cx <= target[2]) & (cy >= target[1]) & (cy <= target[3])
    pool = np.where(inside)[0] if inside.any() else np.arange(len(conf))
    best = int(pool[np.argmax(conf[pool])])
    return [(float(x), float(y), float(c)) for (x, y), c in zip(xy[best], kc[best])]


def load_or_extract_keypoints(video_path, tracker_result, tracks, cfg, out_dir, reuse: bool):
    """keypoints.json in out_dir when it matches this video/stride/pose rate AND these tracks (and --reuse), else
    run the pose model."""
    path = Path(out_dir) / "keypoints.json"
    key = {"video": Path(str(tracker_result.get("video", video_path))).name, "stride": tracker_result.get("stride"),
           "pose_hz": cfg["pool"]["pose_hz"], "proc_size": tracker_result.get("proc_size"),
           "tracks": utils.detections_fingerprint(tracker_result.get("detections"))}
    if reuse and path.exists():
        try:
            data = utils.read_json(path)
            if data.get("key") == key:
                print(f"      reusing pose keypoints from {path}")
                return data["keypoints"]
        except Exception:
            pass
    kps = extract_keypoints(video_path, tracker_result, tracks, cfg)
    utils.write_json(path, {"key": key, "keypoints": {str(k): {str(f): v for f, v in d.items()} for k, d in kps.items()}})
    return kps


# --------------------------------------------------------------------------- the whole pool stage

def run_pool(video_path, tracker_result: dict, tracks: dict, zones: list, cfg: dict, out_dir, reuse: bool = False) -> dict:
    """Stages 2-4 for one video. Returns {"events": [standard events], "dashboard": {...}, "stage3": {...}}."""
    out_dir = Path(out_dir)
    pcfg = cfg["pool"]
    keypoints, pose_note = None, "pose not used"
    if pcfg.get("pose", True) and tracks:
        try:
            keypoints = load_or_extract_keypoints(video_path, tracker_result, tracks, cfg, out_dir, reuse)
            n = sum(len(v) for v in keypoints.values())
            pose_note = f"pose keypoints for {len(keypoints)} swimmer(s), {n} frames"
        except Exception as exc:                       # no model / no video: box-based signals still work
            pose_note = f"pose unavailable ({type(exc).__name__}: {exc}); using box-based arm/head signals"
    print(f"      {pose_note}")

    det_classes = {}
    for row in tracker_result.get("detections") or []:
        det_classes[(int(row[2]), int(row[0]))] = int(row[8]) if len(row) >= 9 else 0
    samples = pool_behaviour.behaviour_samples(tracks, cfg, zones, keypoints, det_classes,
                                               pieces=stitch.piece_spans(tracker_result))
    samples = pool_behaviour.wave_relative_heads(samples, pcfg)       # Rule B (off unless the config turns it on)
    last_seen = {int(tid): float(tr.t[-1]) for tid, tr in tracks.items() if len(tr.t)}
    # lost inside a track, and every join made by stitch.py: judged like the end of a track (see stitch.py)
    gaps = stitch.track_gaps(tracks, float(pcfg["submersion_lost_s"]), tracker_result)
    video_end = float(tracker_result.get("duration_s") or 0)
    stage3 = distress.reason(samples, cfg, last_seen=last_seen, video_end=video_end, gaps=gaps,
                             vanished=vanished_in_water(tracks, tracker_result, pcfg, video_end),
                             same_spot=back_at_same_spot(tracks, tracker_result, pcfg, video_end))

    place_alerts(stage3["events"], tracks, tracker_result, cfg)
    write_stage_files(out_dir, tracker_result, samples, stage3, cfg, gaps)
    events = [to_event(e, tracks, cfg) for e in stage3["events"]]
    for ev in events:                                  # risk curve around the event, for the evidence plot
        curve = stage3["risk"].get(ev["entity_id"], [])
        lo, hi = ev["start_s"] - 15.0, ev["end_s"] + 5.0
        ev["metrics"]["risk_series"] = [p for p in curve if lo <= p[0] <= hi]
    dashboard = build_dashboard(tracker_result, tracks, stage3, cfg, pose_note)
    utils.write_json(out_dir / "pool_dashboard.json", dashboard)
    return {"events": events, "dashboard": dashboard, "stage3": stage3, "samples": samples}


def place_alerts(alerts: list, tracks: dict, tracker_result: dict, cfg: dict) -> None:
    """Add where each alert is, so a lifeguard can act on it: "camera" (config camera.name), "spot" (processing px:
    where a Rule C swimmer vanished, else the centre of their box nearest the alert time) and "area" (part of the
    picture in words). In place."""
    import numpy as np
    width, height = (float(v) for v in (tracker_result.get("proc_size") or (0, 0)))
    camera = str((cfg.get("camera") or {}).get("name") or "").strip()
    for a in alerts:
        if camera:
            a["camera"] = camera
        if "spot" not in a:
            tr = tracks.get(int(a["person_id"]))
            if tr is None or not len(tr.t):
                continue
            k = int(np.argmin(np.abs(np.asarray(tr.t, dtype=float) - float(a["alert_time"]))))
            x1, y1, x2, y2 = (float(v) for v in tr.box[k])
            a["spot"] = [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)]
        area = distress.picture_area(a["spot"][0], a["spot"][1], width, height)
        if area:
            a["area"] = area


def vanished_in_water(tracks: dict, tracker_result: dict, pcfg: dict, video_end: float) -> dict:
    """Rule A (pre-registered, data/ground_truth/PREREGISTRATION_vanish_waves.md): swimmers followed for at least
    vanish_min_seen_s who then disappear while the video goes on for vanish_lost_s more, with their last box at least
    vanish_edge_bh box heights from every picture edge, and no other track starting within vanish_reappear_bh box
    heights of that spot in those seconds (stitch.py already joined the plausible continuations).
    Returns {person_id: {"lost_at": s, "seen_s": s}}; empty when pool.vanish_enabled is off."""
    if not pcfg.get("vanish_enabled"):
        return {}
    import numpy as np
    width, height = (float(v) for v in (tracker_result.get("proc_size") or (0, 0)))
    min_seen, lost_s = float(pcfg["vanish_min_seen_s"]), float(pcfg["vanish_lost_s"])
    edge, near = float(pcfg["vanish_edge_bh"]), float(pcfg["vanish_reappear_bh"])
    starts = [(int(tid), float(tr.t[0]), np.asarray(tr.box[0], dtype=float)) for tid, tr in tracks.items() if len(tr.t)]
    out = {}
    for tid, tr in tracks.items():
        if len(tr.t) < 2:
            continue
        first, last = float(tr.t[0]), float(tr.t[-1])
        if last - first < min_seen or video_end - last < lost_s:
            continue
        x1, y1, x2, y2 = (float(v) for v in tr.box[-1])
        size = max(x2 - x1, y2 - y1, 1.0)
        margin = edge * size
        if x1 < margin or y1 < margin or (width and x2 > width - margin) or (height and y2 > height - margin):
            continue                                              # walked or swam out of the picture
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(other != int(tid) and last < t0 <= last + lost_s
               and np.hypot((b[0] + b[2]) / 2 - cx, (b[1] + b[3]) / 2 - cy) <= near * size
               for other, t0, b in starts):
            continue                                              # someone (probably them) shows up right there
        out[int(tid)] = {"lost_at": round(last, 2), "seen_s": round(last - first, 2)}
    return out


def back_at_same_spot(tracks: dict, tracker_result: dict, pcfg: dict, video_end: float) -> dict:
    """Rule C (pre-registered, data/ground_truth/PREREGISTRATION_same_spot_zones.md): a swimmer followed for at least
    vanish_min_seen_s vanishes with the last box vanish_edge_bh box heights from every picture edge; nobody is seen
    within same_spot_bh box heights of that spot for same_spot_min_s; then somebody (any track: an ID often changes
    after a long gap) is seen there again within same_spot_max_s. Returns {person_id: {"lost_at", "back_at",
    "gap_s"}}; empty when pool.same_spot_enabled is off."""
    if not pcfg.get("same_spot_enabled"):
        return {}
    import numpy as np
    width, height = (float(v) for v in (tracker_result.get("proc_size") or (0, 0)))
    min_seen, edge = float(pcfg["vanish_min_seen_s"]), float(pcfg["vanish_edge_bh"])
    gap_min, gap_max, near = float(pcfg["same_spot_min_s"]), float(pcfg["same_spot_max_s"]), float(pcfg["same_spot_bh"])
    # 'Somebody there' and 'when they vanished' come from ALL detected boxes (the stitched IDs of tracker_result),
    # not only the kept tracks: a small box filtered out by features.min_box_h_px is still somebody at the spot, and
    # the swimmer's own last box may be such a small one (bug hunt 2026-10-11, pool low; bug group B step B6).
    raw = [(int(r[2]), float(r[1]), [float(v) for v in r[3:7]]) for r in tracker_result.get("detections") or []
           if len(r) >= 7]
    if raw:
        rows = [(i, t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for i, t, b in raw]
    else:                                                        # no raw rows (a unit test): the tracks themselves
        rows = [(int(tid), float(t), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                for tid, tr in tracks.items() for t, b in zip(tr.t, np.asarray(tr.box, dtype=float))]
    if not rows:
        return {}
    ids = np.array([r[0] for r in rows])
    when, xs, ys = (np.array([r[k] for r in rows]) for k in (1, 2, 3))
    last_raw: dict[int, tuple[float, list[float]]] = {}
    for i, t, b in raw:
        if i not in last_raw or t > last_raw[i][0]:
            last_raw[i] = (t, b)
    out = {}
    for tid, tr in tracks.items():
        if len(tr.t) < 2:
            continue
        first = float(tr.t[0])
        last, last_box = last_raw.get(int(tid), (float(tr.t[-1]), [float(v) for v in tr.box[-1]]))
        last = max(last, float(tr.t[-1]))
        if last - first < min_seen or video_end - last < gap_min:
            continue
        x1, y1, x2, y2 = (float(v) for v in last_box)
        size = max(x2 - x1, y2 - y1, 1.0)
        margin = edge * size
        if x1 < margin or y1 < margin or (width and x2 > width - margin) or (height and y2 > height - margin):
            continue                                              # left the picture
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        there = (ids != int(tid)) & (np.hypot(xs - cx, ys - cy) <= near * size)
        after = when - last
        if np.any(there & (after > 0) & (after < gap_min)):
            continue                                              # somebody right there soon after: not empty
        back = there & (after >= gap_min) & (after <= gap_max)
        if back.any():
            t_back = float(when[back].min())
            out[int(tid)] = {"lost_at": round(last, 2), "back_at": round(t_back, 2), "gap_s": round(t_back - last, 2),
                             "x": round(cx, 1), "y": round(cy, 1)}
    return out


def write_stage_files(out_dir: Path, tracker_result: dict, samples: list, stage3: dict, cfg: dict,
                      gaps: dict | None = None) -> None:
    """The team's data contract, one file per stage.

    Joined tracks (stitch.py) carry their piece spans in the stage-1 meta line and their gaps in stage2_gaps.json,
    so `pool_behaviour.py --tracks` and `distress.py --behaviour` run on their own give the pipeline's result."""
    state_map = {int(k): str(v) for k, v in (cfg["pool"].get("state_classes") or {}).items()}
    rows = [{"meta": {"fps": tracker_result.get("fps"), "stride": tracker_result.get("stride"),
                      "proc_size": tracker_result.get("proc_size"), "video": tracker_result.get("video"),
                      "stitch_pieces": tracker_result.get("stitch_pieces") or {}}}]
    for r in tracker_result.get("detections") or []:
        cls = int(r[8]) if len(r) >= 9 else 0
        rec = {"person_id": int(r[2]), "timestamp": round(float(r[1]), 3), "frame": int(r[0]),
               "bbox": [round(float(v), 1) for v in r[3:7]], "conf": round(float(r[7]), 3), "class": cls}
        if state_map:
            rec["state"] = state_map.get(cls, "unknown")
        rows.append(rec)
    pool_behaviour.write_jsonl(out_dir / "stage1_tracking.jsonl", rows)
    pool_behaviour.write_jsonl(out_dir / "stage2_behaviour.jsonl", samples)
    utils.write_json(out_dir / "stage2_gaps.json", {str(k): [list(g) for g in v] for k, v in (gaps or {}).items()})
    utils.write_json(out_dir / "stage3_events.json", {"events": stage3["events"], "timeline": stage3["timeline"]})


def to_event(e: dict, tracks: dict, cfg: dict) -> dict:
    """A stage-3 event as a normal standard event dict (DESIGN 3.4 keys), so render/report/UI handle it."""
    import numpy as np
    behavior = {"possible_submersion": "submersion", "early_warning": "aquatic_warning"}.get(e["event"],
                                                                                           "aquatic_distress")
    pid = int(e["person_id"])
    tr = tracks.get(pid)
    start, end = float(e["start_time"]), max(float(e["end_time"]), float(e["start_time"]))
    if end - start < 0.5:
        end = start + 0.5
    conf = 0.85
    if tr is not None and len(tr.t):
        sel = (tr.t >= start - 0.5) & (tr.t <= end + 0.5)
        conf = float(np.mean(tr.conf[sel])) if sel.any() else float(np.mean(tr.conf))
    who = utils.entity_name(cfg, pid, getattr(tr, "cls", 0) if tr is not None else 0)
    checklist = "; ".join(distress.EVIDENCE_TEXT.get(k, k) for k in e["evidence"])
    if behavior == "aquatic_distress":
        evidence = (f"{who} showed a distress pattern from {utils.fmt_time_precise(start)}, held for "
                    f"{float(e['alert_time']) - start:.1f} s before the alert at {utils.fmt_time_precise(e['alert_time'])} "
                    f"in the {e['location']}. Evidence: {checklist}. Risk {e['risk_score']:.0%}.")
    elif behavior == "aquatic_warning" and "back_at_same_spot" in e["evidence"]:
        evidence = (f"{who} vanished in the {e['location']} at {utils.fmt_time_precise(start)}, away from the edges of "
                    f"the picture, and somebody was seen again at the very same spot at "
                    f"{utils.fmt_time_precise(e['alert_time'])} ({float(e['alert_time']) - start:.0f} s later). They may "
                    f"have been under the water: check on them now.")
    elif behavior == "aquatic_warning" and "vanished_in_water" in e["evidence"]:
        evidence = (f"{who} was followed and then vanished in the {e['location']} at {utils.fmt_time_precise(start)}, "
                    f"away from the edges of the picture, and did not come back into view by "
                    f"{utils.fmt_time_precise(e['alert_time'])}. They may be under the water: check on them now.")
    elif behavior == "aquatic_warning":
        evidence = (f"{who} kept showing warning signs from {utils.fmt_time_precise(start)} in the {e['location']} "
                    f"(early warning at {utils.fmt_time_precise(e['alert_time'])}): {checklist}. They may need help: "
                    f"check on them now. Risk {e['risk_score']:.0%}, below the distress alarm level.")
    else:
        evidence = (f"{who} was in trouble and then {('went under' if 'head_submerged' in e['evidence'] else 'was lost from view')} "
                    f"at {utils.fmt_time_precise(start)} in the {e['location']}. Check immediately.")
    snap_t = float(e["alert_time"]) if behavior in ("aquatic_distress", "aquatic_warning") else start
    if tr is not None and len(tr.t):
        snap_t = float(min(max(snap_t, tr.t[0]), tr.t[-1]))
    metrics = {"duration_s": round(end - start, 2), "risk_score": e["risk_score"], "alert_time_s": e["alert_time"],
               "location": e["location"], "evidence_keys": e["evidence"], "peak_time_s": round(snap_t, 2),
               "mean_det_conf": round(conf, 3), "event_type": e["event"], "alert_kind": distress.alert_kind(e),
               "where": distress.where_text(e)}
    for key in ("camera", "area", "spot"):
        if e.get(key):
            metrics[key] = e[key]
    return {"event_id": 0, "entity_id": pid, "behavior": behavior, "start_s": round(start, 2), "end_s": round(end, 2),
            "start": utils.fmt_time(start), "end": utils.fmt_time(end), "duration_s": round(end - start, 2),
            "confidence": round(float(e["risk_score"]), 2),
            "confidence_parts": {"margin": round(float(e["risk_score"]), 2), "duration": 1.0, "detection": round(conf, 2)},
            "evidence": evidence, "metrics": metrics, "zone": e["location"], "baseline": None,
            "snapshot_time_s": round(snap_t, 2), "snapshot": None, "speed_plot": None, "verified": None, "pose": None,
            "severity": "medium" if distress.alert_kind(e) == "aquatic_warning" else "high",
            "entities": [pid], "other_entity": None, "entity_name": who,
            "behavior_name": "Possibly under water" if distress.alert_kind(e) == "under_water"
            else utils.behavior_name(cfg, behavior)}


def build_dashboard(tracker_result: dict, tracks: dict, stage3: dict, cfg: dict, pose_note: str) -> dict:
    """Everything the lifeguard dashboard needs: people, their risk curves, timelines and the alerts."""
    people = []
    for tid in sorted(tracks):
        tr = tracks[tid]
        people.append({"person_id": int(tid), "name": utils.entity_name(cfg, tid, getattr(tr, "cls", 0)),
                       "first_seen": round(float(tr.t[0]), 2), "last_seen": round(float(tr.t[-1]), 2),
                       "final_state": stage3["status"].get(int(tid), "NORMAL")})
    p = cfg["pool"]
    return {"video": tracker_result.get("video"), "duration_s": tracker_result.get("duration_s"),
            "people": people, "risk": {str(k): v for k, v in stage3["risk"].items()},
            "timeline": {str(k): v for k, v in stage3["timeline"].items()}, "alerts": stage3["events"],
            "thresholds": {"watch": p["watch_risk"], "warning": p["warning_risk"], "alert": p["alert_risk"],
                           "mark_spot_s": p["mark_spot_s"]},
            "pose": pose_note}
