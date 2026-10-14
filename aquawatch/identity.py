"""Identity + synchronization record schema for AquaWatch (multi-camera pool CCTV).

A tracker ID ("Track #4") is local to ONE camera's run: it is not a real-world identity, and the same
number in two cameras means nothing. An IdentityRecord keeps that apart:

  - local_track_id : the stitched tracker ID of this camera's run (stage1_tracking.jsonl / events.json);
  - global_id      : a cross-camera identity. It stays None unless a link was VERIFIED (by a person,
                     or by a validated automatic method). Appearance candidates (reid.py) never set it;
  - identity_evidence : candidate links [{camera_id, track_id, score, status: "unverified", ...}];
  - sync           : {offset_s, sync_status, normalized_time} from the camera registry (cameras.py);
  - verification_status : one of the four constants below.

Records are built only from real run files; nothing here invents a camera, track, time or event.
Pure Python (no cv2 / numpy).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cameras as camreg

VERIFIED_MANUAL = "verified_manual"          # a person confirmed the cross-camera link
VERIFIED_AUTOMATED = "verified_automated"    # a validated automatic method confirmed it (none exists yet)
UNVERIFIED = "unverified"                    # candidate links exist, nobody confirmed them
UNAVAILABLE = "unavailable"                  # no cross-camera identity evidence at all
VERIFICATION_STATUSES = (VERIFIED_MANUAL, VERIFIED_AUTOMATED, UNVERIFIED, UNAVAILABLE)
VERIFIED_STATUSES = (VERIFIED_MANUAL, VERIFIED_AUTOMATED)

CANDIDATE_LABEL = "Candidate match - unverified"
NO_IDENTITY_EVIDENCE = "No cross-camera identity evidence."
TRACK_NOTE = "Tracker ID is local to this camera's run; it is not a real-world identity."
LINK_KEYS = ("camera_id", "track_id", "score")


def display_track(track_id: Any, camera_id: Any) -> str:
    """Human text for a local track: 'Track #4 (cam02)'. Never 'person 4'."""
    return f"Track #{int(track_id)} ({camera_id})"


def candidate_text(candidate: Mapping[str, Any]) -> str:
    """'Candidate match - unverified: Track #7 (cam_b), score 0.91' for one candidate link."""
    return (f"{CANDIDATE_LABEL}: {display_track(candidate['track_id'], candidate['camera_id'])}, "
            f"score {float(candidate['score']):.2f}")


def status_for_candidates(candidates: list | None) -> str:
    """Verification status of an automatically linked record: unverified with candidates, else unavailable."""
    return UNVERIFIED if candidates else UNAVAILABLE


def unverified_candidate(candidate: Mapping[str, Any]) -> dict:
    """A copy of an automatic candidate link, checked and marked status 'unverified'.

    Raises ValueError when camera_id / track_id / score is missing, or when the link claims a verified status
    (an automatic candidate is never verified; a verified link is stored with its own evidence instead).
    """
    if not isinstance(candidate, Mapping) or not set(LINK_KEYS) <= set(candidate):
        raise ValueError("each candidate link needs camera_id, track_id and score")
    if candidate.get("status") not in (None, UNVERIFIED):
        raise ValueError(f"an automatic candidate link must be '{UNVERIFIED}', not {candidate.get('status')!r}")
    link = dict(candidate)
    link["track_id"] = int(link["track_id"])
    link["score"] = float(link["score"])
    link["status"] = UNVERIFIED
    link.setdefault("label", CANDIDATE_LABEL)
    return link


@dataclass
class IdentityRecord:
    """One observation of one local track of one camera, with its identity and sync status."""

    camera_id: str
    local_track_id: int
    global_id: str | None = None
    timestamp_s: float | None = None
    wall_time: str | None = None
    bbox: list[float] | None = None
    event: str | None = None
    confidence: float | None = None
    source_video: str | None = None
    source_frame: int | None = None
    evidence: str | None = None
    identity_evidence: list[dict] = field(default_factory=list)
    sync: dict = field(default_factory=lambda: {"offset_s": None, "sync_status": camreg.SYNC_UNAVAILABLE,
                                                "normalized_time": None})
    provenance: dict = field(default_factory=dict)
    verification_status: str = UNAVAILABLE

    def __post_init__(self) -> None:
        """Refuse inconsistent records: a global_id needs a verified status, and a verified status needs one."""
        if self.verification_status not in VERIFICATION_STATUSES:
            raise ValueError(f"verification_status {self.verification_status!r} must be one of "
                             f"{', '.join(VERIFICATION_STATUSES)}")
        if self.global_id is not None and self.verification_status not in VERIFIED_STATUSES:
            raise ValueError("global_id must stay None unless the identity link is verified")
        if self.global_id is None and self.verification_status in VERIFIED_STATUSES:
            raise ValueError(f"status {self.verification_status} needs a global_id")
        for link in self.identity_evidence:
            if not isinstance(link, Mapping) or not set(LINK_KEYS) <= set(link):
                raise ValueError("each identity_evidence link needs camera_id, track_id and score")
        if self.verification_status == UNVERIFIED and not self.identity_evidence:
            raise ValueError("status unverified needs at least one candidate link (else it is unavailable)")
        self.local_track_id = int(self.local_track_id)

    def display(self) -> str:
        """'Track #N (camera X)'."""
        return display_track(self.local_track_id, self.camera_id)

    def to_dict(self) -> dict:
        """Plain JSON-ready dict of every field."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IdentityRecord":
        """Rebuild a record from to_dict() output (unknown keys are ignored)."""
        known = {name for name in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in known})


def attach_candidates(record: IdentityRecord, candidates: list[dict]) -> IdentityRecord:
    """Store automatic candidate links on a record (each marked 'unverified').

    An unverified / unavailable record gets exactly these candidates: status unverified (or unavailable without
    any), global_id None. A VERIFIED record keeps its global_id, status and existing evidence; the new candidates
    are only appended (links to a camera/track already in its evidence are not repeated).
    """
    links = [unverified_candidate(c) for c in candidates or []]
    if record.verification_status in VERIFIED_STATUSES:
        known = {(str(e["camera_id"]), int(e["track_id"])) for e in record.identity_evidence}
        record.identity_evidence = list(record.identity_evidence) + [
            link for link in links if (str(link["camera_id"]), link["track_id"]) not in known]
        return record
    record.identity_evidence = links
    record.global_id = None
    record.verification_status = status_for_candidates(links)
    return record


# ---------------------------------------------------------------------------- reading a run folder

def project_path(path: str | Path) -> str:
    """A path relative to the project root when inside it (forward slashes), else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(camreg.ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_observations(out_dir: str | Path) -> tuple[dict, list[dict], str | None]:
    """(meta, rows, source file name) of a run's per-frame track boxes.

    rows are {"track_id", "t", "frame", "bbox" [x1, y1, x2, y2] in processing pixels, "conf"} from
    stage1_tracking.jsonl (stitched IDs, the ones events.json uses), else from detections.json (raw tracker
    IDs). Returns ({}, [], None) when neither file exists.
    """
    out_dir = Path(out_dir)
    stage1 = out_dir / "stage1_tracking.jsonl"
    if stage1.exists():
        meta, rows = {}, []
        with stage1.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "meta" in row:
                    meta = dict(row["meta"] or {})
                    continue
                if row.get("person_id") is None:
                    continue
                rows.append({"track_id": int(row["person_id"]), "t": float(row["timestamp"]),
                             "frame": int(row["frame"]) if row.get("frame") is not None else None,
                             "bbox": [float(v) for v in row["bbox"]],
                             "conf": float(row["conf"]) if row.get("conf") is not None else None})
        return meta, rows, stage1.name
    detections = out_dir / "detections.json"
    if detections.exists():
        document = json.loads(detections.read_text(encoding="utf-8"))
        rows = [{"track_id": int(r[2]), "t": float(r[1]), "frame": int(r[0]), "bbox": [float(v) for v in r[3:7]],
                 "conf": float(r[7]) if len(r) > 7 and r[7] is not None else None}
                for r in document.get("detections") or []]
        return dict(document.get("meta") or {}), rows, detections.name
    return {}, [], None


def read_events(out_dir: str | Path) -> dict:
    """events.json of a run folder as a dict ({} when missing)."""
    path = Path(out_dir) / "events.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _snapshot(out_dir: Path, event: Mapping[str, Any]) -> str | None:
    """Project path of an event's snapshot image when the file really exists, else None."""
    snapshot = event.get("snapshot") or event.get("snapshot_path")
    if not snapshot:
        return None
    path = Path(str(snapshot))
    path = path if path.is_absolute() else out_dir / path
    return project_path(path) if path.is_file() else None


def _sync(camera: Mapping[str, Any], seconds: Any) -> dict:
    """The sync block of a record: offset, status and the normalized time (None when unavailable)."""
    info = camreg.sync_info(dict(camera))
    normalized, status = camreg.normalized_time(dict(camera), seconds)
    return {"offset_s": info["offset_s"], "sync_status": status, "normalized_time": normalized}


def _wall_time(camera: Mapping[str, Any], seconds: Any) -> str | None:
    """Camera-declared wall clock of a moment (None when the camera has no start_time)."""
    if seconds is None or camera.get("start_time") in (None, ""):
        return None
    return camreg.wall_clock(dict(camera), seconds)


def _video(camera: Mapping[str, Any], meta: Mapping[str, Any], document: Mapping[str, Any]) -> str | None:
    """The source video named by the registry, else by the run's own metadata."""
    value = camera.get("video") or meta.get("video") or (document.get("meta") or {}).get("video")
    return str(value).replace("\\", "/") if value else None


def track_records(camera: Mapping[str, Any], out_dir: str | Path) -> list[IdentityRecord]:
    """One IdentityRecord per local track of a run folder (first observation, mean detector confidence).

    event / evidence come from the track's most confident events.json event (None when it has none).
    verification_status is UNAVAILABLE: a single run carries no cross-camera identity evidence.
    """
    out_dir = Path(out_dir)
    meta, rows, source = read_observations(out_dir)
    document = read_events(out_dir)
    events = [e for e in document.get("events") or [] if isinstance(e, Mapping)]
    camera_id = str(camera["id"])
    video = _video(camera, meta, document)
    by_track: dict[int, list[dict]] = {}
    for row in rows:
        by_track.setdefault(row["track_id"], []).append(row)
    records = []
    for track_id in sorted(by_track):
        track = sorted(by_track[track_id], key=lambda r: (r["t"], r["frame"] or 0))
        first = track[0]
        confs = [r["conf"] for r in track if r["conf"] is not None]
        own = [e for e in events if _entity(e) == track_id]
        best = max(own, key=lambda e: float(e.get("confidence") or 0.0), default=None)
        records.append(IdentityRecord(
            camera_id=camera_id,
            local_track_id=track_id,
            timestamp_s=round(first["t"], 3),
            wall_time=_wall_time(camera, first["t"]),
            bbox=[round(v, 1) for v in first["bbox"]],
            event=str(best.get("behavior")) if best else None,
            confidence=round(sum(confs) / len(confs), 3) if confs else None,
            source_video=video,
            source_frame=first["frame"],
            evidence=_snapshot(out_dir, best) if best else None,
            sync=_sync(camera, first["t"]),
            provenance={"source": source, "run_dir": project_path(out_dir),
                        "producer": "tracker.py + stitch.py (YOLO + ByteTrack)",
                        "track_span_s": [round(first["t"], 3), round(track[-1]["t"], 3)],
                        "n_boxes": len(track), "confidence_kind": "mean detector confidence of the track's boxes",
                        "event_ids": [e.get("event_id") for e in own], "note": TRACK_NOTE},
            verification_status=UNAVAILABLE,
        ))
    return records


def _entity(event: Mapping[str, Any]) -> int | None:
    """The track ID an events.json event is about (entity_id, else person_id)."""
    value = event.get("entity_id") if event.get("entity_id") is not None else event.get("person_id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def table_row(record: IdentityRecord | Mapping[str, Any]) -> dict:
    """An IdentityRecord (or its dict) -> one row dict for event_index.EventIndex.set_identities().

    Unverified links of the record go to `candidates`; links carrying a verified status go to
    `identity_evidence`. first_s / last_s come from the record time and provenance.track_span_s.
    """
    data = record.to_dict() if isinstance(record, IdentityRecord) else dict(record)
    links = list(data.get("identity_evidence") or [])
    span = (data.get("provenance") or {}).get("track_span_s") or [data.get("timestamp_s"), data.get("timestamp_s")]
    return {"local_track_id": int(data["local_track_id"]), "global_id": data.get("global_id"),
            "verification_status": data.get("verification_status", UNAVAILABLE),
            "identity_evidence": [link for link in links if link.get("status") in VERIFIED_STATUSES],
            "candidates": [link for link in links if link.get("status") not in VERIFIED_STATUSES],
            "first_s": span[0], "last_s": span[1], "first_wall": data.get("wall_time"), "last_wall": None,
            "provenance": dict(data.get("provenance") or {})}


def reid_identity_rows(result: Mapping[str, Any], camera_list: list[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """reid.run_reid() output -> {camera_id: [rows for EventIndex.set_identities]} for every embedded camera.

    Every row has global_id None and status unverified (with candidates) or unavailable (without). Cameras whose
    Re-ID failed (an `error` in result["cameras"]) get no rows, so their stored identities are left alone.
    """
    cams = {str(c["id"]): c for c in camera_list}
    ok = {str(row["camera_id"]): row for row in result.get("cameras") or [] if not row.get("error")}
    rows: dict[str, list[dict]] = {camera_id: [] for camera_id in ok}
    for track in result.get("tracks") or []:
        camera_id = str(track["camera_id"])
        if camera_id not in ok:
            continue
        camera = cams.get(camera_id, {})
        candidates = [unverified_candidate(c) for c in track.get("candidates") or []]
        rows[camera_id].append({
            "local_track_id": int(track["track_id"]), "global_id": None,
            "verification_status": status_for_candidates(candidates),
            "identity_evidence": [], "candidates": candidates,
            "first_s": track.get("first_s"), "last_s": track.get("last_s"),
            "first_wall": _wall_time(camera, track.get("first_s")), "last_wall": _wall_time(camera, track.get("last_s")),
            "provenance": {"producer": "reid.py", "method": result.get("method"), "generated": result.get("generated"),
                           "run_dir": ok[camera_id].get("run_dir"), "n_crops": track.get("n_crops", 0),
                           "note": track.get("note") or TRACK_NOTE, "disclaimer": result.get("disclaimer")},
        })
    return rows


def event_records(camera: Mapping[str, Any], out_dir: str | Path) -> list[IdentityRecord]:
    """One IdentityRecord per events.json event: event start time, the track's box nearest to it, snapshot."""
    out_dir = Path(out_dir)
    meta, rows, source = read_observations(out_dir)
    document = read_events(out_dir)
    camera_id = str(camera["id"])
    video = _video(camera, meta, document)
    records = []
    for event in document.get("events") or []:
        track_id = _entity(event)
        if track_id is None:
            continue
        start = float(event.get("start_s", 0.0))
        own = [r for r in rows if r["track_id"] == track_id]
        nearest = min(own, key=lambda r: abs(r["t"] - start), default=None)
        metrics = event.get("metrics") if isinstance(event.get("metrics"), Mapping) else {}
        records.append(IdentityRecord(
            camera_id=camera_id,
            local_track_id=track_id,
            timestamp_s=round(start, 3),
            wall_time=_wall_time(camera, start),
            bbox=[round(v, 1) for v in nearest["bbox"]] if nearest else None,
            event=str(event.get("behavior", "unknown")),
            confidence=float(event["confidence"]) if event.get("confidence") is not None else None,
            source_video=video,
            source_frame=nearest["frame"] if nearest else None,
            evidence=_snapshot(out_dir, event),
            sync=_sync(camera, start),
            provenance={"source": "events.json", "run_dir": project_path(out_dir), "event_id": event.get("event_id"),
                        "end_s": event.get("end_s"), "alert_time_s": metrics.get("alert_time_s"),
                        "evidence_text": event.get("evidence"), "box_source": source,
                        "box_time_s": round(nearest["t"], 3) if nearest else None, "note": TRACK_NOTE},
            verification_status=UNAVAILABLE,
        ))
    return records
