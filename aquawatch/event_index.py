"""Multi-camera event index for the swimming-pool CCTV intelligence pipeline.

The existing detector and behavior engine remain unchanged. This module converts
those results into a camera-aware, queryable record and stores clarification
references separately so a later natural-language query can reuse them.

Tables (SQLite, parameterised SQL only):
- events     : one row per event of one camera (seconds into the video + wall-clock ISO times)
- cameras    : one row per registered camera (name, video, wall clock of its first frame, run stats, and the
               registry's synchronization: offset_s + sync_status, see cameras.py)
- tracks     : one row per track of one camera (time span, box size, position, peak risk, final pool state,
               and an `appearance_json` placeholder for the Stage 2 appearance embeddings)
- identities : one row per local track of one camera with its cross-camera identity status:
               global_id (NULL unless verified), verification_status (identity.VERIFICATION_STATUSES),
               identity_evidence (JSON list: the evidence behind a VERIFIED status), candidates (JSON list of
               ranked automatic candidate links, always status "unverified"), first/last seen in seconds and
               wall time, provenance (JSON). Written by set_identities() (ingest.py --reid stores reid.py
               candidates here); read by identities() and candidates_for().
- links      : generic track-to-track links that come with their OWN evidence and status (for example
               geometry.py links between overlapping views). Written by store_links(), read by links().
- meta       : key -> JSON value (site name, dataset label), set_meta() / get_meta().

Wall-clock times are stored as fixed-width ISO text 'YYYY-MM-DDTHH:MM:SS.mmm' (see cameras.format_wall),
so they compare correctly as strings. `index_run` replaces the events / tracks / cameras rows of a camera,
`set_identities` replaces its identity rows (verified ones are kept) and `store_links` upserts, so re-ingesting
is idempotent. index_run also drops the camera's links, geometry identity groups and unverified identity rows:
stitching runs again on every run.py (also with --reuse), so track numbers can change. Re-run geometry.py (and
ingest --reid) after an ingest; manually / automatically VERIFIED identity rows are kept.
"""
from __future__ import annotations

import json
import re
import sqlite3
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import identity
from cameras import ROOT, format_wall, normalize_wall, parse_wall, sync_info

# columns added to the original events table (older databases are migrated with ALTER TABLE)
EVENT_EXTRA_COLUMNS = {
    "start_wall": "TEXT",
    "end_wall": "TEXT",
    "alert_s": "REAL",
    "alert_wall": "TEXT",
    "location": "TEXT",
    "severity": "TEXT",
    "details_json": "TEXT",
}
# columns added to the original cameras table (synchronization from the camera registry)
CAMERA_EXTRA_COLUMNS = {
    "offset_s": "REAL",
    "sync_status": "TEXT",
}
# statuses a stored link may carry ("unavailable" means "no evidence", which is not a link)
LINK_STATUSES = (identity.VERIFIED_MANUAL, identity.VERIFIED_AUTOMATED, identity.UNVERIFIED)


def _wall(start_time: datetime | None, seconds: Any) -> str | None:
    """Wall-clock ISO text of `seconds` after `start_time` (None when either is missing)."""
    if start_time is None or seconds is None:
        return None
    return format_wall(start_time + timedelta(seconds=float(seconds)))


def _bound(value: Any) -> str | None:
    """A since/until argument (datetime or text) -> the stored ISO format, or None."""
    return None if value in (None, "") else normalize_wall(value)


def _project_path(path: Path) -> str:
    """A path relative to the project root when it is inside it (forward slashes), else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_track_boxes(out_dir: str | Path) -> dict[int, list[tuple[float, list[float]]]]:
    """{track_id: [(t, [x1, y1, x2, y2]), ...]} from stage1_tracking.jsonl, else detections.json.

    stage1_tracking.jsonl has the joined (stitched) IDs that events.json uses; detections.json only the
    raw tracker IDs. Returns {} when neither file is there.
    """
    out_dir = Path(out_dir)
    boxes: dict[int, list[tuple[float, list[float]]]] = {}
    stage1 = out_dir / "stage1_tracking.jsonl"
    if stage1.exists():
        with stage1.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "meta" in row or row.get("person_id") is None:
                    continue
                boxes.setdefault(int(row["person_id"]), []).append(
                    (float(row["timestamp"]), [float(v) for v in row["bbox"]]))
        return boxes
    detections = out_dir / "detections.json"
    if detections.exists():
        document = json.loads(detections.read_text(encoding="utf-8"))
        for row in document.get("detections") or []:           # [frame, t, id, x1, y1, x2, y2, conf, cls]
            boxes.setdefault(int(row[2]), []).append((float(row[1]), [float(v) for v in row[3:7]]))
    return boxes


def summarize_tracks(boxes: Mapping[int, list], dashboard: Mapping[str, Any] | None = None) -> list[dict]:
    """One summary dict per track: span, box count, median box w/h, mean centre, peak risk, final state."""
    dashboard = dashboard or {}
    people = {int(p["person_id"]): p for p in dashboard.get("people") or [] if p.get("person_id") is not None}
    risk = dashboard.get("risk") or {}
    tracks = []
    for track_id in sorted(boxes):
        rows = sorted(boxes[track_id], key=lambda item: item[0])
        widths = [b[2] - b[0] for _, b in rows]
        heights = [b[3] - b[1] for _, b in rows]
        series = risk.get(str(track_id)) or []
        tracks.append({
            "track_id": int(track_id),
            "first_s": rows[0][0],
            "last_s": rows[-1][0],
            "n_boxes": len(rows),
            "median_w": round(statistics.median(widths), 2),
            "median_h": round(statistics.median(heights), 2),
            "mean_cx": round(sum((b[0] + b[2]) / 2.0 for _, b in rows) / len(rows), 2),
            "mean_cy": round(sum((b[1] + b[3]) / 2.0 for _, b in rows) / len(rows), 2),
            "peak_risk": max((float(r) for _, r in series), default=None) if dashboard else None,
            "final_state": (people.get(track_id) or {}).get("final_state"),
            "swimmer": (1 if track_id in people else 0) if dashboard else None,
        })
    return tracks


class EventIndex:
    """A SQLite index containing events from one or more cameras."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    person_id INTEGER,
                    behavior TEXT NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    confidence REAL,
                    evidence TEXT,
                    snapshot TEXT,
                    metadata_json TEXT,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (camera_id, event_id)
                )
                """
            )
            # migrate databases made before the wall-clock columns existed
            existing = {row["name"] for row in connection.execute("PRAGMA table_info(events)")}
            for column, kind in EVENT_EXTRA_COLUMNS.items():
                if column not in existing:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {column} {kind}")   # names are constants
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_camera_person ON events(camera_id, person_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_behavior_time ON events(behavior, start_s)"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_wall ON events(start_wall, end_wall)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    video TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    preset TEXT,
                    zones TEXT,
                    out_dir TEXT,
                    duration_s REAL,
                    end_time TEXT,
                    fps REAL,
                    proc_w INTEGER,
                    proc_h INTEGER,
                    n_swimmers INTEGER,
                    n_events INTEGER,
                    n_alerts INTEGER,
                    indexed_at TEXT
                )
                """
            )
            existing = {row["name"] for row in connection.execute("PRAGMA table_info(cameras)")}
            for column, kind in CAMERA_EXTRA_COLUMNS.items():          # migrate databases without sync columns
                if column not in existing:
                    connection.execute(f"ALTER TABLE cameras ADD COLUMN {column} {kind}")  # names are constants
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    camera_id TEXT NOT NULL,
                    local_track_id INTEGER NOT NULL,
                    global_id TEXT,
                    verification_status TEXT NOT NULL,
                    identity_evidence_json TEXT NOT NULL,
                    candidates_json TEXT NOT NULL,
                    first_s REAL,
                    last_s REAL,
                    first_wall TEXT,
                    last_wall TEXT,
                    provenance_json TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (camera_id, local_track_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_identities_global ON identities(global_id)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS links (
                    camera_a TEXT NOT NULL,
                    track_a INTEGER NOT NULL,
                    camera_b TEXT NOT NULL,
                    track_b INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    score REAL,
                    evidence_json TEXT NOT NULL,
                    stored_at TEXT NOT NULL,
                    PRIMARY KEY (camera_a, track_a, camera_b, track_b, method)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_links_b ON links(camera_b, track_b)")
            connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value_json TEXT)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    first_s REAL NOT NULL,
                    last_s REAL NOT NULL,
                    first_wall TEXT,
                    last_wall TEXT,
                    n_boxes INTEGER NOT NULL,
                    median_w REAL,
                    median_h REAL,
                    mean_cx REAL,
                    mean_cy REAL,
                    peak_risk REAL,
                    final_state TEXT,
                    swimmer INTEGER,
                    appearance_json TEXT,
                    PRIMARY KEY (camera_id, track_id)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_tracks_wall ON tracks(first_wall, last_wall)")
            connection.commit()
        finally:
            connection.close()

    # ------------------------------------------------------------------ writing

    @staticmethod
    def _event_row(event: Mapping[str, Any], position: int, camera_id: str, video_path: str | Path,
                   metadata_text: str, start_time: datetime | None, indexed_at: str) -> tuple:
        """One events-table row (column order of _EVENT_COLUMNS) from a standard event dict."""
        metrics = event.get("metrics") if isinstance(event.get("metrics"), Mapping) else {}
        alert_s = metrics.get("alert_time_s")
        details = {key: value for key, value in metrics.items() if key != "risk_series"}
        for key in ("behavior_name", "entity_name", "zone", "snapshot_time_s"):
            if event.get(key) is not None:
                details[key] = event[key]
        return (
            int(event.get("event_id", position)),
            str(camera_id),
            str(video_path),
            event.get("entity_id") if event.get("entity_id") is not None else event.get("person_id"),
            str(event.get("behavior", "unknown")),
            float(event.get("start_s", 0.0)),
            float(event.get("end_s", 0.0)),
            float(event.get("confidence", 0.0)) if event.get("confidence") is not None else None,
            str(event.get("evidence", "")),
            event.get("snapshot") or event.get("snapshot_path"),
            metadata_text,
            event.get("indexed_at") or indexed_at,
            _wall(start_time, event.get("start_s", 0.0)),
            _wall(start_time, event.get("end_s", 0.0)),
            float(alert_s) if alert_s is not None else None,
            _wall(start_time, alert_s),
            metrics.get("location") or event.get("zone"),
            event.get("severity"),
            json.dumps(details, sort_keys=True, default=str) if details else None,
        )

    _EVENT_COLUMNS = ("event_id", "camera_id", "video_path", "person_id", "behavior", "start_s", "end_s",
                      "confidence", "evidence", "snapshot", "metadata_json", "indexed_at",
                      "start_wall", "end_wall", "alert_s", "alert_wall", "location", "severity", "details_json")

    def _insert_events(self, connection: sqlite3.Connection, rows: list[tuple]) -> None:
        """Upsert event rows inside an open connection."""
        columns = ", ".join(self._EVENT_COLUMNS)
        marks = ", ".join("?" for _ in self._EVENT_COLUMNS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in self._EVENT_COLUMNS if c not in ("camera_id", "event_id"))
        connection.executemany(
            f"INSERT INTO events ({columns}) VALUES ({marks}) "
            f"ON CONFLICT(camera_id, event_id) DO UPDATE SET {updates}",
            rows,
        )

    def index_events(
        self,
        camera_id: str,
        video_path: str | Path,
        events: Iterable[Mapping[str, Any]],
        metadata: Mapping[str, Any] | None = None,
        start_time: str | datetime | None = None,
    ) -> int:
        """Insert or replace events for one camera and return the inserted count.

        With `start_time` (wall clock of the video's first frame) the wall-clock columns are filled too.
        """
        start = parse_wall(start_time) if start_time not in (None, "") else None
        metadata_text = json.dumps(dict(metadata or {}), sort_keys=True, default=str)
        indexed_at = format_wall(datetime.now())
        rows = []
        for event in events:
            rows.append(self._event_row(event, len(rows) + 1, camera_id, video_path, metadata_text, start, indexed_at))
        if not rows:
            return 0
        connection = self._connect()
        try:
            self._insert_events(connection, rows)
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    @staticmethod
    def _track_rows(camera_id: str, tracks: Iterable[Mapping[str, Any]], start: datetime | None) -> list[tuple]:
        """tracks-table rows from summarize_tracks() dicts."""
        rows = []
        for track in tracks:
            appearance = track.get("appearance")
            rows.append((
                str(camera_id), int(track["track_id"]), float(track["first_s"]), float(track["last_s"]),
                _wall(start, track["first_s"]), _wall(start, track["last_s"]), int(track["n_boxes"]),
                track.get("median_w"), track.get("median_h"), track.get("mean_cx"), track.get("mean_cy"),
                track.get("peak_risk"), track.get("final_state"), track.get("swimmer"),
                json.dumps(appearance) if appearance is not None else None,
            ))
        return rows

    @staticmethod
    def _insert_tracks(connection: sqlite3.Connection, rows: list[tuple]) -> None:
        """Insert-or-replace track rows inside an open connection."""
        connection.executemany(
            """
            INSERT OR REPLACE INTO tracks (
                camera_id, track_id, first_s, last_s, first_wall, last_wall, n_boxes,
                median_w, median_h, mean_cx, mean_cy, peak_risk, final_state, swimmer, appearance_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def index_tracks(self, camera_id: str, tracks: Iterable[Mapping[str, Any]],
                     start_time: str | datetime | None = None) -> int:
        """Insert or replace track summaries (see summarize_tracks) for one camera; returns the count."""
        start = parse_wall(start_time) if start_time not in (None, "") else None
        rows = self._track_rows(camera_id, tracks, start)
        connection = self._connect()
        try:
            self._insert_tracks(connection, rows)
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    def set_appearance(self, camera_id: str, track_id: int, appearance: Any) -> None:
        """Store a Stage 2 appearance descriptor (any JSON value) for one track."""
        connection = self._connect()
        try:
            connection.execute("UPDATE tracks SET appearance_json = ? WHERE camera_id = ? AND track_id = ?",
                               (json.dumps(appearance), str(camera_id), int(track_id)))
            connection.commit()
        finally:
            connection.close()

    def delete_camera(self, camera_id: str) -> None:
        """Remove every row (camera, events, tracks, identities, links on either side) of one camera."""
        connection = self._connect()
        try:
            for table in ("events", "tracks", "cameras", "identities"):    # table names are constants
                connection.execute(f"DELETE FROM {table} WHERE camera_id = ?", (str(camera_id),))
            connection.execute("DELETE FROM links WHERE camera_a = ? OR camera_b = ?", (str(camera_id),) * 2)
            connection.commit()
        finally:
            connection.close()

    def index_run(self, camera: Mapping[str, Any], out_dir: str | Path) -> dict[str, Any]:
        """Index one camera's run folder (events.json, pool_dashboard.json, stage1_tracking.jsonl or detections.json).

        `camera` is a cameras.load_cameras() dict. All earlier rows of this camera are replaced in one
        transaction, so re-ingesting the same camera never duplicates rows. Returns the camera's stats:
        {"camera_id", "swimmers", "tracks", "events", "alerts", "duration_s", "start_wall", "end_wall"};
        swimmers / alerts come from pool_dashboard.json (people / alerts) when the run has one.
        Raises FileNotFoundError when events.json is missing.
        """
        out_dir = Path(out_dir)
        events_path = out_dir / "events.json"
        if not events_path.exists():
            raise FileNotFoundError(f"{events_path} not found")
        document = json.loads(events_path.read_text(encoding="utf-8"))
        meta = document.get("meta") or {}
        dashboard_path = out_dir / "pool_dashboard.json"
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8")) if dashboard_path.exists() else None

        camera_id = str(camera["id"])
        start = parse_wall(camera["start_time"])
        video = str(camera.get("video") or meta.get("video") or "")
        indexed_at = format_wall(datetime.now())

        events = []
        for event in document.get("events") or []:
            event = dict(event)
            snapshot = event.get("snapshot")
            if snapshot and not Path(str(snapshot)).is_absolute():       # keep it findable from the project root
                event["snapshot"] = _project_path(out_dir / str(snapshot))
            events.append(event)
        metadata = {"source": _project_path(events_path), "scenario": meta.get("scenario"),
                    "duration_s": meta.get("duration_s"), "fps": meta.get("fps")}
        metadata_text = json.dumps(metadata, sort_keys=True, default=str)
        event_rows = [self._event_row(e, i + 1, camera_id, video, metadata_text, start, indexed_at)
                      for i, e in enumerate(events)]
        tracks = summarize_tracks(read_track_boxes(out_dir), dashboard)
        track_rows = self._track_rows(camera_id, tracks, start)

        if dashboard is not None:
            swimmers = len(dashboard.get("people") or [])
        else:
            swimmers = int(meta.get("n_people") or meta.get("n_tracks") or len(tracks))
        if dashboard is not None:                                      # lifeguard alerts of the pool stage
            alerts = len(dashboard.get("alerts") or [])
        else:
            alerts = sum(1 for e in events if str(e.get("severity", "")).lower() == "high")
        duration = meta.get("duration_s") if meta.get("duration_s") is not None else (dashboard or {}).get("duration_s")
        proc = meta.get("proc_size") or [None, None]
        sync = sync_info(dict(camera))
        stats = {"camera_id": camera_id, "swimmers": swimmers, "tracks": len(tracks), "events": len(events),
                 "alerts": alerts, "duration_s": duration, "start_wall": format_wall(start),
                 "end_wall": _wall(start, duration), "offset_s": sync["offset_s"], "sync_status": sync["sync_status"]}

        connection = self._connect()
        try:
            for table in ("events", "tracks", "cameras"):                  # replace: drop this camera's old rows
                connection.execute(f"DELETE FROM {table} WHERE camera_id = ?", (camera_id,))
            # Stitching runs again on every run.py (also with --reuse), so this camera's track numbers may have
            # changed: its links, identity groups and Re-ID candidates no longer point at the right tracks. They are
            # dropped here; geometry.py (and ingest --reid) rebuild them (bug hunt 2026-10-11, multicam medium).
            connection.execute("DELETE FROM links WHERE camera_a = ? OR camera_b = ?", (camera_id, camera_id))
            placeholders = ", ".join("?" for _ in identity.VERIFIED_STATUSES)   # a person's verification stays
            connection.execute(f"DELETE FROM identities WHERE camera_id = ? AND verification_status NOT IN "
                               f"({placeholders})", (camera_id, *identity.VERIFIED_STATUSES))
            connection.execute(
                """
                INSERT INTO cameras (camera_id, name, video, start_time, preset, zones, out_dir, duration_s,
                                     end_time, fps, proc_w, proc_h, n_swimmers, n_events, n_alerts, indexed_at,
                                     offset_s, sync_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (camera_id, str(camera.get("name") or camera_id), video, format_wall(start), camera.get("preset"),
                 camera.get("zones"), _project_path(out_dir), duration, stats["end_wall"], meta.get("fps"),
                 proc[0], proc[1], swimmers, len(events), alerts, indexed_at, sync["offset_s"], sync["sync_status"]),
            )
            if event_rows:
                self._insert_events(connection, event_rows)
            if track_rows:
                self._insert_tracks(connection, track_rows)
            connection.commit()
        finally:
            connection.close()
        self._forget_camera_geometry(camera_id)
        return stats

    # ------------------------------------------------------------------ reading

    def query(
        self,
        camera_id: str | None = None,
        person_id: int | None = None,
        behavior: str | None = None,
        since_s: float | None = None,
        until_s: float | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return matching event records in chronological order."""
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if person_id is not None:
            clauses.append("person_id = ?")
            params.append(int(person_id))
        if behavior:
            clauses.append("behavior = ?")
            params.append(behavior.lower())
        if since_s is not None:
            clauses.append("start_s >= ?")
            params.append(float(since_s))
        if until_s is not None:
            clauses.append("end_s <= ?")
            params.append(float(until_s))

        sql = "SELECT * FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY start_s, camera_id, event_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))

        connection = self._connect()
        try:
            rows = connection.execute(sql, params).fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            connection.close()

    def events_between(
        self,
        since_wall: str | datetime | None = None,
        until_wall: str | datetime | None = None,
        camera_id: str | None = None,
        behavior: str | None = None,
        person_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Events that overlap the wall-clock window [since_wall, until_wall] (either end may be open).

        An event overlaps when it ends at or after `since_wall` and starts at or before `until_wall`.
        Events indexed without a camera start time have no wall times and only match an open window.
        """
        clauses: list[str] = []
        params: list[Any] = []
        since, until = _bound(since_wall), _bound(until_wall)
        if since is not None:
            clauses.append("end_wall >= ?")
            params.append(since)
        if until is not None:
            clauses.append("start_wall <= ?")
            params.append(until)
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if behavior:
            clauses.append("behavior = ?")
            params.append(behavior.lower())
        if person_id is not None:
            clauses.append("person_id = ?")
            params.append(int(person_id))
        sql = "SELECT * FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY start_wall, camera_id, start_s, event_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        connection = self._connect()
        try:
            return [self._row_to_dict(row) for row in connection.execute(sql, params).fetchall()]
        finally:
            connection.close()

    def query_tracks(
        self,
        camera_id: str | None = None,
        since_wall: str | datetime | None = None,
        until_wall: str | datetime | None = None,
        track_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Tracks seen during the wall-clock window [since_wall, until_wall] (overlap; either end may be open)."""
        clauses: list[str] = []
        params: list[Any] = []
        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if track_id is not None:
            clauses.append("track_id = ?")
            params.append(int(track_id))
        since, until = _bound(since_wall), _bound(until_wall)
        if since is not None:
            clauses.append("last_wall >= ?")
            params.append(since)
        if until is not None:
            clauses.append("first_wall <= ?")
            params.append(until)
        sql = "SELECT * FROM tracks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY first_wall, camera_id, track_id"
        connection = self._connect()
        try:
            rows = []
            for row in connection.execute(sql, params).fetchall():
                record = dict(row)
                text = record.pop("appearance_json", None)
                record["appearance"] = json.loads(text) if text else None
                rows.append(record)
            return rows
        finally:
            connection.close()

    def list_cameras(self) -> list[dict[str, Any]]:
        """All indexed cameras with their run stats, ordered by id."""
        connection = self._connect()
        try:
            return [dict(row) for row in connection.execute("SELECT * FROM cameras ORDER BY camera_id").fetchall()]
        finally:
            connection.close()

    # ------------------------------------------------------------------ identities (one row per local track)

    _IDENTITY_COLUMNS = ("camera_id", "local_track_id", "global_id", "verification_status", "identity_evidence_json",
                         "candidates_json", "first_s", "last_s", "first_wall", "last_wall", "provenance_json",
                         "updated_at")

    @staticmethod
    def _identity_row(camera_id: str, record: Any, updated_at: str) -> tuple:
        """Check one identity record and return its table row (column order of _IDENTITY_COLUMNS).

        `record` is an identity.IdentityRecord or a mapping with local_track_id (or track_id), global_id,
        verification_status, identity_evidence (list), candidates (list), first_s, last_s, first_wall,
        last_wall, provenance (dict). Raises ValueError for anything that would claim more than its evidence:
        a global_id without a verified status, a verified status without global_id or evidence, a candidate
        that is not "unverified", "unverified" without any link, "unavailable" with candidates.
        """
        if isinstance(record, identity.IdentityRecord):
            data = identity.table_row(record)
        else:
            data = dict(record)
        track = data.get("local_track_id", data.get("track_id"))
        if track is None:
            raise ValueError("identity record needs local_track_id")
        track = int(track)
        status = str(data.get("verification_status") or identity.UNAVAILABLE)
        global_id = data.get("global_id")
        global_id = None if global_id in (None, "") else str(global_id)
        evidence = list(data.get("identity_evidence") or [])
        candidates = [identity.unverified_candidate(c) for c in data.get("candidates") or []]
        where = f"identity of {identity.display_track(track, camera_id)}"
        if status not in identity.VERIFICATION_STATUSES:
            raise ValueError(f"{where}: verification_status {status!r} must be one of "
                             f"{', '.join(identity.VERIFICATION_STATUSES)}")
        if global_id is not None and status not in identity.VERIFIED_STATUSES:
            raise ValueError(f"{where}: global_id must stay null unless the identity is verified")
        if status in identity.VERIFIED_STATUSES and (global_id is None or not evidence):
            raise ValueError(f"{where}: status {status} needs a global_id and its identity_evidence")
        if status == identity.UNVERIFIED and not (candidates or evidence):
            raise ValueError(f"{where}: status unverified needs at least one candidate or evidence item")
        if status == identity.UNAVAILABLE and (candidates or evidence):
            raise ValueError(f"{where}: status unavailable cannot carry candidates or evidence")
        if any(str(c["camera_id"]) == str(camera_id) and c["track_id"] == track for c in candidates):
            raise ValueError(f"{where}: a track cannot be its own candidate")
        number = lambda v: None if v is None else float(v)                   # noqa: E731
        return (str(camera_id), track, global_id, status, json.dumps(evidence, sort_keys=True, default=str),
                json.dumps(candidates, sort_keys=True, default=str), number(data.get("first_s")),
                number(data.get("last_s")), data.get("first_wall"), data.get("last_wall"),
                json.dumps(dict(data.get("provenance") or {}), sort_keys=True, default=str), updated_at)

    @staticmethod
    def _identity_dict(row: sqlite3.Row) -> dict[str, Any]:
        """An identities row with its JSON columns decoded and a 'Track #N (camera)' display text."""
        result = dict(row)
        result["identity_evidence"] = json.loads(result.pop("identity_evidence_json") or "[]")
        result["candidates"] = json.loads(result.pop("candidates_json") or "[]")
        result["provenance"] = json.loads(result.pop("provenance_json") or "{}")
        result["display"] = identity.display_track(result["local_track_id"], result["camera_id"])
        return result

    def set_identities(self, camera_id: str, records: Iterable[Any], keep_verified: bool = True) -> int:
        """Replace the identity rows of one camera (one transaction); returns the number of rows written.

        Every record is checked by _identity_row (nothing is stored that claims more than its evidence).
        With keep_verified (default) a row already stored as verified_manual / verified_automated is never
        downgraded by an automatic re-run: its global_id, status and identity_evidence are kept, the new
        candidates / time span are taken, and a verified row whose track is absent from `records` is kept.
        Calling it twice with the same records gives the same table (idempotent).
        """
        now = format_wall(datetime.now())
        rows = [self._identity_row(camera_id, record, now) for record in records]
        tracks = [row[1] for row in rows]
        if len(set(tracks)) != len(tracks):
            raise ValueError(f"camera {camera_id}: a local track appears twice in the identity records")
        columns = ", ".join(self._IDENTITY_COLUMNS)
        marks = ", ".join("?" for _ in self._IDENTITY_COLUMNS)
        connection = self._connect()
        try:
            kept = {}
            if keep_verified:
                placeholders = ", ".join("?" for _ in identity.VERIFIED_STATUSES)
                for old in connection.execute(
                        f"SELECT * FROM identities WHERE camera_id = ? AND verification_status IN ({placeholders})",
                        (str(camera_id), *identity.VERIFIED_STATUSES)).fetchall():
                    kept[int(old["local_track_id"])] = old
            merged = []
            for row in rows:
                old = kept.pop(row[1], None)
                if old is not None and row[3] not in identity.VERIFIED_STATUSES:
                    row = (row[0], row[1], old["global_id"], old["verification_status"],
                           old["identity_evidence_json"]) + row[5:]
                merged.append(row)
            merged += [tuple(old[c] for c in self._IDENTITY_COLUMNS) for old in kept.values()]
            connection.execute("DELETE FROM identities WHERE camera_id = ?", (str(camera_id),))
            connection.executemany(f"INSERT INTO identities ({columns}) VALUES ({marks})", merged)
            connection.commit()
            return len(merged)
        finally:
            connection.close()

    def identities(self, camera_id: str | None = None, track_id: int | None = None, status: str | None = None,
                   global_id: str | None = None) -> list[dict[str, Any]]:
        """Stored identity rows (decoded), ordered by camera and track; every filter is optional."""
        clauses, params = [], []
        for column, value in (("camera_id", camera_id), ("local_track_id", track_id),
                              ("verification_status", status), ("global_id", global_id)):
            if value is not None:
                clauses.append(f"{column} = ?")                            # column names are constants
                params.append(int(value) if column == "local_track_id" else str(value))
        sql = "SELECT * FROM identities"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY camera_id, local_track_id"
        connection = self._connect()
        try:
            return [self._identity_dict(row) for row in connection.execute(sql, params).fetchall()]
        finally:
            connection.close()

    def candidates_for(self, camera_id: str, track_id: int) -> list[dict[str, Any]]:
        """Ranked candidate links of one local track (each status "unverified"); [] when none are stored."""
        rows = self.identities(camera_id=camera_id, track_id=track_id)
        return rows[0]["candidates"] if rows else []

    # ------------------------------------------------------------------ links (each with its own evidence)

    @staticmethod
    def _link_row(link: Mapping[str, Any], stored_at: str) -> tuple:
        """Check one link and return its row (sides in canonical order, so a<->b is stored once).

        A link is {camera_a, track_a, camera_b, track_b, status, method, evidence, score (optional)}.
        status must be verified_manual, verified_automated or unverified; method a non-empty name; evidence a
        non-empty text / list / dict (stored as JSON). Raises ValueError otherwise, or for a track linked to itself.
        """
        try:
            side_a = (str(link["camera_a"]), int(link["track_a"]))
            side_b = (str(link["camera_b"]), int(link["track_b"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"link needs camera_a, track_a, camera_b, track_b ({exc})") from exc
        if not side_a[0] or not side_b[0]:
            raise ValueError("link camera ids must not be empty")
        if side_a == side_b:
            raise ValueError(f"link from {identity.display_track(side_a[1], side_a[0])} to itself")
        status = str(link.get("status") or "")
        if status not in LINK_STATUSES:
            raise ValueError(f"link status {status!r} must be one of {', '.join(LINK_STATUSES)}")
        method = str(link.get("method") or "").strip()
        if not method:
            raise ValueError("link needs a method name")
        evidence = link.get("evidence")
        if evidence is None or (isinstance(evidence, (str, list, dict)) and not evidence):
            raise ValueError("link needs its evidence (a link without evidence is not stored)")
        score = link.get("score")
        if side_b < side_a:
            side_a, side_b = side_b, side_a
        return (side_a[0], side_a[1], side_b[0], side_b[1], method, status,
                None if score is None else float(score), json.dumps(evidence, sort_keys=True, default=str), stored_at)

    def store_links(self, links: Iterable[Mapping[str, Any]], replace: bool = False,
                    methods: Iterable[str] = ()) -> int:
        """Upsert cross-camera links that come with their own evidence + status; returns the count stored.

        Input: an iterable of {camera_a, track_a, camera_b, track_b, status, method, evidence, score}, e.g. the
        links of geometry.py. Every link is checked first (see _link_row); one bad link stores nothing.
        The key is (both sides in canonical order, method): storing the same link again updates it. With
        replace=True every stored link of the methods present in `links` AND of `methods` is deleted first, so a
        re-run of one method leaves no stale links, even when it finds none (pass methods=[...]); other methods are
        untouched.
        """
        now = format_wall(datetime.now())
        rows = [self._link_row(link, now) for link in links]
        connection = self._connect()
        try:
            if replace:
                for method in sorted(set(methods) | {row[4] for row in rows}):
                    connection.execute("DELETE FROM links WHERE method = ?", (method,))
            connection.executemany(
                "INSERT INTO links (camera_a, track_a, camera_b, track_b, method, status, score, evidence_json, "
                "stored_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(camera_a, track_a, camera_b, track_b, method) DO UPDATE SET status=excluded.status, "
                "score=excluded.score, evidence_json=excluded.evidence_json, stored_at=excluded.stored_at",
                rows,
            )
            connection.commit()
            return len(rows)
        finally:
            connection.close()

    def _forget_camera_geometry(self, camera_id: str) -> None:
        """Drop the geometry identity groups and same-swimmer events that involve this camera (see index_run)."""
        groups = self.get_meta("geometry_global_ids")
        if groups:
            self.set_meta("geometry_global_ids", {gid: members for gid, members in groups.items()
                                                  if not any(m[0] == camera_id for m in members)})
        same = self.get_meta("geometry_same_swimmer_events")
        if same:
            self.set_meta("geometry_same_swimmer_events",
                          [s for s in same if camera_id not in (s.get("cameras") or [])])

    def links(self, camera_id: str | None = None, track_id: int | None = None, status: str | None = None,
              method: str | None = None) -> list[dict[str, Any]]:
        """Stored links (evidence decoded). camera_id / track_id match either side of a link."""
        clauses, params = [], []
        if camera_id is not None and track_id is not None:
            clauses.append("((camera_a = ? AND track_a = ?) OR (camera_b = ? AND track_b = ?))")
            params += [str(camera_id), int(track_id)] * 2
        elif camera_id is not None:
            clauses.append("(camera_a = ? OR camera_b = ?)")
            params += [str(camera_id)] * 2
        elif track_id is not None:
            raise ValueError("track_id needs camera_id (tracker IDs are local to one camera)")
        if status is not None:
            clauses.append("status = ?")
            params.append(str(status))
        if method is not None:
            clauses.append("method = ?")
            params.append(str(method))
        sql = "SELECT * FROM links"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY camera_a, track_a, camera_b, track_b, method"
        connection = self._connect()
        try:
            rows = []
            for row in connection.execute(sql, params).fetchall():
                record = dict(row)
                record["evidence"] = json.loads(record.pop("evidence_json") or "null")
                record["display_a"] = identity.display_track(record["track_a"], record["camera_a"])
                record["display_b"] = identity.display_track(record["track_b"], record["camera_b"])
                rows.append(record)
            return rows
        finally:
            connection.close()

    # ------------------------------------------------------------------ meta (site name, dataset label)

    def set_meta(self, key: str, value: Any) -> None:
        """Store one JSON value under a key (replaces the old value)."""
        connection = self._connect()
        try:
            connection.execute("INSERT OR REPLACE INTO meta (key, value_json) VALUES (?, ?)",
                               (str(key), json.dumps(value, default=str)))
            connection.commit()
        finally:
            connection.close()

    def get_meta(self, key: str, default: Any = None) -> Any:
        """The JSON value stored under a key, else `default`."""
        connection = self._connect()
        try:
            row = connection.execute("SELECT value_json FROM meta WHERE key = ?", (str(key),)).fetchone()
            return json.loads(row["value_json"]) if row else default
        finally:
            connection.close()

    def index_file(
        self,
        camera_id: str,
        video_path: str | Path,
        events_path: str | Path,
    ) -> int:
        """Index an existing events.json document without rerunning detection."""
        document = json.loads(Path(events_path).read_text(encoding="utf-8"))
        events = document.get("events", [])
        metadata = document.get("meta", {})
        metadata.setdefault("source", str(events_path))
        return self.index_events(camera_id, video_path, events, metadata)

    def query_text(self, text: str, limit: int | None = 20) -> list[dict[str, Any]]:
        """Simple natural-language search over camera, person, behavior and evidence.

        This is intentionally conservative: it understands fixed terms such as
        "camera 2", "person 7", "drowning", and "last 10 minutes". Semantic
        retrieval can be layered on top without changing the indexed schema.
        """
        normalized = text.lower()
        camera_id = None
        person_id = None
        behavior = None
        since_s = None
        if "camera" in normalized:
            match = re.search(r"camera\s+([0-9]+)", normalized)
            if match:
                camera_id = f"camera_{match.group(1)}"
        match = re.search(r"person\s+([0-9]+)", normalized)
        if match:
            person_id = int(match.group(1))
        for value in ("drowning", "distress", "fall", "loitering", "running", "crowding", "zone_intrusion"):
            if value in normalized:
                behavior = value
                break
        match = re.search(r"last\s+(\d+)\s*(?:minutes?|mins?)", normalized)
        if match:
            since_s = max(0.0, float(match.group(1)) * 60.0)
        return self.query(camera_id, person_id, behavior, since_s, limit=limit)

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["person_id"] = result.get("person_id")
        result["snapshot"] = result.get("snapshot")
        result["metadata"] = json.loads(result.pop("metadata_json", None) or "{}")
        details = result.pop("details_json", None)
        result["details"] = json.loads(details) if details else {}
        return result


class ClarifyMemory:
    """Persistent, camera-scoped references supplied through a clarification."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}
        else:
            self._data = {}

    def store(
        self,
        camera_id: str,
        reference: str,
        aliases: Mapping[str, str] | None = None,
    ) -> None:
        camera = self._data.setdefault(str(camera_id), {})
        camera["reference"] = str(reference)
        if aliases:
            camera.update({str(key): str(value) for key, value in aliases.items()})
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")

    def resolve(self, camera_id: str, alias: str) -> str | None:
        camera = self._data.get(str(camera_id), {})
        normalized_alias = re.sub(r"[^a-z0-9]+", "_", str(alias).strip().lower()).strip("_")
        return camera.get(normalized_alias) or camera.get(str(alias))

    def get(self, camera_id: str) -> dict[str, str]:
        return dict(self._data.get(str(camera_id), {}))


def main() -> None:
    """Command-line helper: index an existing events.json file."""
    import argparse

    parser = argparse.ArgumentParser(description="Index an existing event report for a camera")
    parser.add_argument("--database", required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    index = EventIndex(args.database)
    count = index.index_file(args.camera, args.video, args.events)
    print(f"Indexed {count} events for {args.camera}")


if __name__ == "__main__":
    main()
