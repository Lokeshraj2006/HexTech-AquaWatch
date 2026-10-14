"""AquaWatch multi-camera ingest: run the pool pipeline for every registered camera and index the results.

    python ingest.py --cameras samples/multicam/cameras.yaml --db outputs/multicam/aquawatch.sqlite3

For each camera of the registry (cameras.py) it runs

    run.py --video <video> --scenario <preset> [--zones <zones>] --out outputs/multicam/<id> --reuse

as a subprocess (same Python), then indexes that folder (events.json, pool_dashboard.json, track spans) into the
SQLite event index with wall-clock times (camera start_time + seconds). A camera whose run or indexing fails is
reported and skipped; the ingest itself never crashes. Re-ingesting a camera replaces its rows.

    python ingest.py --cameras data/virtual_cctv/cameras.yaml --db outputs/virtual_cctv/aquawatch.sqlite3
                     --out-root outputs/virtual_cctv --index-only --reid

With --reid, after indexing, reid.py ranks appearance candidates across the registry's run folders
(<out-root>/<camera id>) and stores them in the index's identities table (EventIndex.set_identities): every
candidate has status "unverified" and global_id stays null. A camera whose Re-ID fails is reported; its stored
identities are left as they were.

With --openvocab, the open-vocabulary CLIP index (openvocab.py: track crops + frames) is built next to the index,
so Ask the Video can find described things ("a child in a blue life jacket"). Unchanged cameras are reused.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cameras as camreg
import identity
from event_index import EventIndex

ROOT = camreg.ROOT
DEFAULT_DB = "outputs/multicam/aquawatch.sqlite3"
DEFAULT_OUT_ROOT = "outputs/multicam"


def build_command(camera: dict, out_dir: Path, python: str | None = None, no_video: bool = False,
                  no_privacy: bool = False, privacy: bool = False) -> list[str]:
    """The run.py command line for one camera (contract item 2). Faces are blurred only with privacy=True."""
    command = [python or sys.executable, "-B", str(Path(__file__).resolve().parent / "run.py"),
               "--video", str(camera["video_path"]), "--scenario", camera["preset"]]
    if camera.get("zones_path"):
        command += ["--zones", str(camera["zones_path"])]
    command += ["--out", str(out_dir), "--reuse"]
    if camera.get("name"):
        command += ["--camera-name", str(camera["name"])]   # every alert and spoken line names the camera
    if no_video:
        command.append("--no-video")
    if privacy:
        command.append("--privacy")
    elif no_privacy:
        command.append("--no-privacy")
    return command


def _last_line(text: str) -> str:
    """Last non-empty line of a process output, ASCII only, for a one-line status."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    return tail.encode("ascii", "replace").decode("ascii")


def run_camera(camera: dict, out_dir: Path, python: str | None = None, no_video: bool = False,
               no_privacy: bool = False, privacy: bool = False,
               timeout_s: float | None = None) -> dict:
    """Run run.py for one camera. Returns {"ok", "returncode", "seconds", "message", "log"} (never raises)."""
    command = build_command(camera, out_dir, python, no_video, no_privacy, privacy)
    started = time.time()
    try:
        proc = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": None, "seconds": time.time() - started,
                "message": f"timed out after {timeout_s} s", "log": ""}
    except OSError as exc:
        return {"ok": False, "returncode": None, "seconds": time.time() - started,
                "message": f"could not start run.py ({exc})", "log": ""}
    log = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == 0
    message = "run ok" if ok else f"run.py exit {proc.returncode}: {_last_line(proc.stderr) or _last_line(proc.stdout)}"
    return {"ok": ok, "returncode": proc.returncode, "seconds": time.time() - started, "message": message, "log": log}


def store_reid(index: EventIndex, camera_list: list[dict], out_root: str | Path, settings: dict | None = None,
               dataset_label: str | None = None, echo=print) -> dict:
    """Run reid.py on <out_root>/<id> for these cameras and store the candidates as identity rows.

    Returns {"result": run_reid output, "stored": {camera_id: rows written}, "errors": {camera_id: message}}.
    Rows are built by identity.reid_identity_rows (global_id None, status unverified / unavailable).
    """
    import reid                                                       # numpy + cv2: only needed with --reid
    result = reid.run_reid(camera_list, out_root, settings, echo=echo, dataset_label=dataset_label)
    errors = {str(row["camera_id"]): row["error"] for row in result["cameras"] if row.get("error")}
    stored = {}
    for camera_id, rows in identity.reid_identity_rows(result, camera_list).items():
        stored[camera_id] = index.set_identities(camera_id, rows)
    return {"result": result, "stored": stored, "errors": errors}


def ingest(camera_list: list[dict], db_path: str | Path, out_root: str | Path = DEFAULT_OUT_ROOT,
           python: str | None = None, no_video: bool = False, index_only: bool = False,
           timeout_s: float | None = None, runner=None, echo=print, reid: bool = False,
           reid_cameras: list[dict] | None = None, reid_settings: dict | None = None,
           site: str | None = None, dataset_label: str | None = None, no_privacy: bool = False,
           privacy: bool = False) -> list[dict]:
    """Run + index every camera; returns one result dict per camera (status, swimmers, events, alerts, ...).

    reid=True then runs Re-ID over `reid_cameras` (default: camera_list) minus the cameras that failed here,
    stores their candidates (store_reid) and adds "identities" (rows stored) and "reid_status" to the results
    of camera_list. site / dataset_label are saved in the index's meta table when given.
    """
    runner = runner or run_camera
    index = EventIndex(camreg.resolve_path(db_path))
    out_root = camreg.resolve_path(out_root)
    if site:
        index.set_meta("site", site)
    if dataset_label:
        index.set_meta("dataset_label", dataset_label)
    results = []
    for number, camera in enumerate(camera_list, start=1):
        out_dir = out_root / camera["id"]
        prefix = f"[{number}/{len(camera_list)}] {camera['id']}"
        result = {"camera_id": camera["id"], "name": camera["name"], "ok": False, "status": "",
                  "swimmers": None, "events": None, "alerts": None, "start_wall": camera["start_wall"],
                  "end_wall": None, "out_dir": out_dir}
        if not index_only:
            if not Path(camera["video_path"]).exists():
                result["status"] = f"FAILED: video {camera['video']} not found"
                echo(f"{prefix}: {result['status']} - skipped")
                results.append(result)
                continue
            if not (out_dir / "detections.json").exists():
                echo(f"{prefix}: no cached detections in {out_dir}; run.py will run the detector (slow)")
            try:
                extra = ({"privacy": True} if privacy else {"no_privacy": True} if no_privacy else {})  # only when asked
                run = runner(camera, out_dir, python=python, no_video=no_video, timeout_s=timeout_s, **extra)
            except Exception as exc:                                     # a broken runner must not stop the ingest
                run = {"ok": False, "seconds": 0.0, "message": f"run failed ({type(exc).__name__}: {exc})"}
            if not run.get("ok"):
                result["status"] = "FAILED: " + str(run.get("message", "run failed"))
                echo(f"{prefix}: {result['status']} - skipped")
                results.append(result)
                continue
            run_note = f"run ok ({float(run.get('seconds') or 0.0):.1f} s)"
        else:
            run_note = "index only"
        try:
            stats = index.index_run(camera, out_dir)
        except Exception as exc:                                         # missing / corrupt output files
            result["status"] = f"FAILED: indexing ({type(exc).__name__}: {exc})"
            echo(f"{prefix}: {result['status']} - skipped")
            results.append(result)
            continue
        result.update(ok=True, status="ok", swimmers=stats["swimmers"], events=stats["events"],
                      alerts=stats["alerts"], end_wall=stats["end_wall"])
        echo(f"{prefix}: {run_note}, indexed {stats['swimmers']} swimmers, {stats['tracks']} tracks, "
             f"{stats['events']} events, {stats['alerts']} alerts")
        results.append(result)
    if reid:
        failed_here = {r["camera_id"] for r in results if not r["ok"]}
        cams = [c for c in (reid_cameras if reid_cameras is not None else camera_list) if c["id"] not in failed_here]
        echo(f"Re-ID candidates (unverified) for {len(cams)} camera(s):")
        try:
            outcome = store_reid(index, cams, out_root, reid_settings, dataset_label, echo)
        except Exception as exc:                                         # Re-ID must never crash the ingest
            outcome = {"stored": {}, "errors": {c["id"]: f"{type(exc).__name__}: {exc}" for c in cams}}
        for result in results:
            camera_id = result["camera_id"]
            if camera_id in outcome["stored"]:
                result.update(identities=outcome["stored"][camera_id], reid_status="ok")
            elif camera_id in outcome["errors"]:
                result.update(identities=None, reid_status="FAILED: " + str(outcome["errors"][camera_id]))
            else:
                result.update(identities=None, reid_status="skipped (not ingested)")
        for camera_id, count in sorted(outcome["stored"].items()):
            echo(f"  {camera_id}: {count} identity rows stored (global_id null, candidates unverified)")
        for camera_id, error in sorted(outcome["errors"].items()):
            echo(f"  {camera_id}: Re-ID FAILED ({error}); stored identities unchanged".encode("ascii", "replace")
                 .decode("ascii"))
    return results


def _clock(wall: str | None) -> str:
    """'2026-10-09T14:00:21.500' -> '14:00:21' ('-' when unknown)."""
    return wall[11:19] if wall and len(wall) >= 19 else "-"


def summary_table(results: list[dict]) -> str:
    """Plain ASCII table: camera, name, swimmers, events, alerts, wall-clock span, status (+ identities with --reid)."""
    with_reid = any("reid_status" in r for r in results)
    header = ("camera", "name", "swimmers", "events", "alerts", "from", "to", "status") + (
        ("identities", "re-id") if with_reid else ())
    rows = [header]
    for r in results:
        row = (r["camera_id"], r["name"], "-" if r["swimmers"] is None else str(r["swimmers"]),
               "-" if r["events"] is None else str(r["events"]),
               "-" if r["alerts"] is None else str(r["alerts"]),
               _clock(r["start_wall"]), _clock(r["end_wall"]), r["status"])
        if with_reid:
            row += ("-" if r.get("identities") is None else str(r["identities"]), str(r.get("reid_status") or "-"))
        rows.append(row)
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    lines = []
    for position, row in enumerate(rows):
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if position == 0:
            lines.append("  ".join("-" * w for w in widths))
    text = "\n".join(lines)
    return text.encode("ascii", "replace").decode("ascii")


def main(argv=None) -> int:
    """CLI entry point. Exit code 0 when every camera was ingested, 1 when some failed, 2 for a bad registry."""
    parser = argparse.ArgumentParser(description="AquaWatch: run the pool pipeline on every camera and index it")
    parser.add_argument("--cameras", required=True, metavar="cameras.yaml", help="camera registry")
    parser.add_argument("--db", default=DEFAULT_DB, metavar="PATH", help=f"SQLite index (default {DEFAULT_DB})")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"run folders go to <out-root>/<camera id> (default {DEFAULT_OUT_ROOT})")
    parser.add_argument("--only", nargs="+", metavar="ID", help="ingest only these camera ids")
    parser.add_argument("--index-only", action="store_true", help="skip run.py, only index existing run folders")
    parser.add_argument("--no-video", action="store_true", help="pass --no-video to run.py (faster)")
    parser.add_argument("--privacy", action="store_true", help="blur faces in the camera outputs")
    parser.add_argument("--no-privacy", action="store_true", help="do not blur faces (the default)")
    parser.add_argument("--timeout", type=float, metavar="S", help="give up on one camera's run after S seconds")
    parser.add_argument("--reid", action="store_true",
                        help="after indexing, store unverified appearance Re-ID candidates (reid.py) in the index")
    parser.add_argument("--openvocab", action="store_true",
                        help="after indexing, build the open-vocabulary CLIP index next to --db (openvocab.py)")
    args = parser.parse_args(argv)
    import cpu_limits
    cpu_limits.apply()                  # below-normal priority + thread budget for this run

    try:
        registry = camreg.load_registry(args.cameras)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    camera_list = registry["cameras"]
    if args.only:
        unknown = sorted(set(args.only) - {c["id"] for c in camera_list})
        if unknown:
            print(f"ERROR: unknown camera id(s): {', '.join(unknown)}")
            return 2
        camera_list = [c for c in camera_list if c["id"] in set(args.only)]

    print(f"AquaWatch ingest: {registry['site'] or args.cameras}, {len(camera_list)} camera(s) -> {args.db}")
    if registry.get("label"):
        print(f"DATASET: {registry['label']}".encode("ascii", "replace").decode("ascii"))
    results = ingest(camera_list, args.db, args.out_root, no_video=args.no_video, no_privacy=args.no_privacy,
                     privacy=args.privacy,
                     index_only=args.index_only,
                     timeout_s=args.timeout, reid=args.reid, reid_cameras=registry["cameras"],
                     site=registry.get("site") or None, dataset_label=registry.get("label"))
    print()
    print(summary_table(results))
    failed = [r for r in results if not r["ok"]]
    print()
    print(f"{len(results) - len(failed)} of {len(results)} camera(s) indexed into {args.db}")
    if args.reid:
        reid_failed = [r for r in results if r["ok"] and r.get("reid_status") != "ok"]
        print(f"{len(results) - len(failed) - len(reid_failed)} of {len(results) - len(failed)} camera(s) got "
              f"Re-ID candidates (all unverified; global_id stays null)")
        failed += reid_failed
    if args.openvocab:
        import openvocab                                  # torch / open_clip: only needed with --openvocab
        try:
            openvocab.build(args.db)
        except Exception as exc:                          # optional step: report, do not crash the ingest
            print(f"open-vocabulary index FAILED: {exc}".encode("ascii", "replace").decode("ascii"))
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
