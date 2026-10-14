"""Compare the events AquaWatch found with hand-written ground-truth labels (DESIGN 3.10).

    python evaluate.py --labels labels.csv --outputs outputs/        # each outputs/<clip_stem>/events.json
    python evaluate.py --labels labels.csv --events outputs/x/events.json --clip x.mp4

labels.csv columns: clip, entity_description, behavior, start_s, end_s
  * one row per true incident; behavior is one of loitering | zone_intrusion | running | fall | crowding | near_miss
  * a row with behavior `none` says "this clip has no incidents at all" (start_s / end_s left empty)
  * times are seconds ("12.5") or mm:ss ("0:12")

How a prediction is matched to a label (per clip and per behaviour):
  1. widen the label by +/- tolerance seconds (default 2),
  2. a prediction and a label are a candidate pair if their time intervals overlap,
  3. pairs are taken greedily, largest overlap first, each label and each prediction used once.
Matched pairs are true positives (TP); unmatched predictions are false positives (FP); unmatched labels are
false negatives (FN). Entity IDs are NOT compared, because labels describe people by appearance, not by ID.

Pure Python + utils (no OpenCV / torch), so it can be unit-tested anywhere.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import utils

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
# Friendly spellings accepted in labels.csv.
BEHAVIOR_ALIASES = {"zone": "zone_intrusion", "intrusion": "zone_intrusion", "run": "running",
                    "loiter": "loitering", "fell": "fall", "person_down": "fall", "crowd": "crowding",
                    "crowded": "crowding", "nearmiss": "near_miss", "near_collision": "near_miss",
                    "no_event": "none", "no_events": "none", "nothing": "none"}
REQUIRED_COLUMNS = ("clip", "behavior", "start_s", "end_s")
EPS = 1e-9


class LabelError(ValueError):
    """The labels file (or an events file) cannot be understood. The message says where."""


# --------------------------------------------------------------------------- parsing helpers

def clip_key(name) -> str:
    """Name used to pair labels with outputs: file name without folders or video extension.

    'samples/x.mp4' -> 'x', 'x' -> 'x'. Names with dots ('clip.v2') keep them unless a video extension follows.
    """
    base = Path(str(name).strip().replace("\\", "/")).name
    suffix = Path(base).suffix.lower()
    return base[: -len(suffix)] if suffix in VIDEO_EXTENSIONS else base


def parse_time(text) -> float | None:
    """Parse '12.5', '12', '1:05', '0:01:05' or '1:05.5' into seconds. Empty text gives None."""
    if text is None:
        return None
    original = text
    if isinstance(text, (int, float)):
        value = float(text)
    else:
        text = str(text).strip()
        if not text:
            return None
        try:
            parts = [float(p) for p in text.split(":")]
        except ValueError:
            raise ValueError(f"'{text}' is not a time (use seconds like 12.5 or mm:ss like 0:12)") from None
        if len(parts) > 3:
            raise ValueError(f"'{text}' is not a time (use seconds like 12.5 or mm:ss like 0:12)")
        value = 0.0
        for part in parts:                                  # h:m:s -> ((h * 60) + m) * 60 + s
            value = value * 60.0 + part
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"time '{original}' must be a finite number of seconds, 0 or more")
    return value


def normalise_behavior(text: str) -> str:
    """'Zone Intrusion' -> 'zone_intrusion'; also maps a few aliases. Raises ValueError if unknown."""
    key = str(text).strip().lower().replace("-", "_").replace(" ", "_")
    key = BEHAVIOR_ALIASES.get(key, key)
    if key != "none" and key not in utils.BEHAVIORS:
        allowed = ", ".join(utils.BEHAVIORS + ("none",))
        raise ValueError(f"unknown behavior '{text}' (use one of: {allowed})")
    return key


def load_labels(path) -> list[dict]:
    """Read labels.csv into [{clip, entity_description, behavior, start_s, end_s}].

    Blank lines and lines starting with '#' are ignored. `none` rows have start_s = end_s = None.
    Raises LabelError with the line number for anything malformed.
    """
    path = Path(path)
    if not path.exists():
        raise LabelError(f"labels file not found: {path}")
    labels: list[dict] = []
    header: list[str] | None = None
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or all(not cell.strip() for cell in row) or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [cell.strip().lower() for cell in row]
                missing = [c for c in REQUIRED_COLUMNS if c not in header]
                if missing:
                    raise LabelError(f"{path.name}: header is missing column(s) {missing}; "
                                     "expected clip,entity_description,behavior,start_s,end_s")
                continue
            line = reader.line_num
            rec = {name: (row[i].strip() if i < len(row) else "") for i, name in enumerate(header)}
            try:
                behavior = normalise_behavior(rec.get("behavior", ""))
                if not rec.get("clip"):
                    raise ValueError("clip is empty")
                start = end = None
                if behavior != "none":
                    start, end = parse_time(rec.get("start_s")), parse_time(rec.get("end_s"))
                    if start is None or end is None:
                        raise ValueError("start_s and end_s are required unless behavior is none")
                    if end < start:
                        raise ValueError(f"end_s ({end:g}) is before start_s ({start:g})")
            except ValueError as exc:
                raise LabelError(f"{path.name} line {line}: {exc}") from None
            labels.append({"clip": rec["clip"], "entity_description": rec.get("entity_description", ""),
                           "behavior": behavior, "start_s": start, "end_s": end})
    if header is None:
        raise LabelError(f"{path.name} is empty")
    return labels


# Early warnings ("swimmer may need help") ask the lifeguard to look; they are not incidents with a start and an end,
# so they are neither matched against labels nor counted as false alarms.
ADVISORY_BEHAVIORS = {"aquatic_warning"}


def load_events(path) -> tuple[str | None, list[dict]]:
    """Read an events.json written by report.py. Returns (video path stored in it, list of events)."""
    path = Path(path)
    if not path.exists():
        raise LabelError(f"events file not found: {path}")
    data = utils.read_json(path)
    if not isinstance(data, dict) or "events" not in data:
        raise LabelError(f"{path} does not look like an events.json (no 'events' list)")
    events = []
    for ev in data["events"]:
        if ev.get("behavior") in ADVISORY_BEHAVIORS:    # a heads-up for the lifeguard, not an incident: not scored
            continue
        try:
            events.append({"event_id": ev.get("event_id"), "entity_id": ev.get("entity_id"),
                           "behavior": str(ev["behavior"]), "start_s": float(ev["start_s"]),
                           "end_s": float(ev["end_s"])})
        except (KeyError, TypeError, ValueError):
            raise LabelError(f"{path}: an event is missing behavior / start_s / end_s") from None
    return data.get("video"), events


# --------------------------------------------------------------------------- matching

def overlap_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap of [a0, a1] and [b0, b1]; negative = gap size; 0 = they just touch."""
    return min(a1, b1) - max(a0, b0)


def match_intervals(labels: list[dict], preds: list[dict], tolerance: float = 2.0):
    """Greedy one-to-one matching of ONE behaviour's labels and predictions in ONE clip.

    Returns (pairs, unmatched_label_indices, unmatched_pred_indices) where pairs is
    [(label_index, pred_index, overlap_s), ...]. Each label is widened by +/- tolerance seconds first.
    Candidates are taken largest overlap first (ties: smaller start error, then list order).
    """
    candidates = []
    for i, lab in enumerate(labels):
        lo, hi = lab["start_s"] - tolerance, lab["end_s"] + tolerance
        for j, pred in enumerate(preds):
            ov = overlap_seconds(lo, hi, pred["start_s"], pred["end_s"])
            if ov >= -EPS:                                  # touching counts as overlapping
                candidates.append((-ov, abs(pred["start_s"] - lab["start_s"]), i, j))
    candidates.sort()
    used_labels, used_preds, pairs = set(), set(), []
    for neg_ov, _start_err, i, j in candidates:
        if i in used_labels or j in used_preds:
            continue
        used_labels.add(i)
        used_preds.add(j)
        pairs.append((i, j, max(0.0, -neg_ov)))
    pairs.sort()
    return (pairs,
            [i for i in range(len(labels)) if i not in used_labels],
            [j for j in range(len(preds)) if j not in used_preds])


def safe_mean(values) -> float | None:
    """Mean of a list, or None when it is empty."""
    return float(sum(values) / len(values)) if values else None


def prf(tp: int, fp: int, fn: int) -> dict:
    """TP / FP / FN -> precision, recall, F1. A ratio with nothing to divide by is None (undefined, not 0)."""
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_clip(labels: list[dict], preds: list[dict], tolerance: float = 2.0) -> dict:
    """Match one clip's labels (no `none` rows) with its predictions, behaviour by behaviour.

    Returns {"matches": [...], "missed": [label dicts], "false_positives": [prediction dicts]}.
    """
    behaviors = list(dict.fromkeys([x["behavior"] for x in labels] + [p["behavior"] for p in preds]))
    matches, missed, false_pos = [], [], []
    for beh in behaviors:
        lab_b = [x for x in labels if x["behavior"] == beh]
        pred_b = [p for p in preds if p["behavior"] == beh]
        pairs, miss_idx, fp_idx = match_intervals(lab_b, pred_b, tolerance)
        for i, j, ov in pairs:
            lab, pred = lab_b[i], pred_b[j]
            matches.append({"behavior": beh, "label": lab, "prediction": pred, "overlap_s": ov,
                            "start_error_s": pred["start_s"] - lab["start_s"],
                            "end_error_s": pred["end_s"] - lab["end_s"]})
        missed.extend(lab_b[i] for i in miss_idx)
        false_pos.extend(pred_b[j] for j in fp_idx)
    return {"matches": matches, "missed": missed, "false_positives": false_pos}


def evaluate(labels: list[dict], predictions_by_clip: dict, tolerance: float = 2.0) -> dict:
    """Score all labelled clips.

    labels               rows from load_labels()
    predictions_by_clip  {clip_key: [event dicts]}; a labelled clip missing from it is skipped (not evaluated)
    Returns a JSON-friendly report dict (see format_report for the readable version).
    """
    by_clip: dict[str, dict] = {}                           # clip_key -> {"name", "labels": [...], "none": bool}
    warnings: list[str] = []
    for row in labels:
        entry = by_clip.setdefault(clip_key(row["clip"]), {"name": row["clip"], "rows": []})
        entry["rows"].append(row)

    per_behavior = {b: {"tp": 0, "fp": 0, "fn": 0, "start_errors": [], "end_errors": []} for b in utils.BEHAVIORS}
    clips_report, skipped = {}, []
    matches_out, missed_out, fp_out, none_details = [], [], [], []
    none_clips = 0

    for key, entry in by_clip.items():
        real = [r for r in entry["rows"] if r["behavior"] != "none"]
        none_rows = [r for r in entry["rows"] if r["behavior"] == "none"]
        if real and none_rows:
            warnings.append(f"clip '{entry['name']}' has both 'none' and real labels; the 'none' row is ignored")
        is_none_clip = bool(none_rows) and not real
        if key not in predictions_by_clip:
            skipped.append(entry["name"])
            continue
        preds = predictions_by_clip[key]
        result = evaluate_clip(real, preds, tolerance)
        none_clips += is_none_clip

        for beh in {m["behavior"] for m in result["matches"]} | {x["behavior"] for x in result["missed"]} | \
                {p["behavior"] for p in result["false_positives"]}:
            per_behavior.setdefault(beh, {"tp": 0, "fp": 0, "fn": 0, "start_errors": [], "end_errors": []})
        for m in result["matches"]:
            slot = per_behavior[m["behavior"]]
            slot["tp"] += 1
            slot["start_errors"].append(abs(m["start_error_s"]))
            slot["end_errors"].append(abs(m["end_error_s"]))
            matches_out.append({"clip": entry["name"], "behavior": m["behavior"],
                                "label_start_s": m["label"]["start_s"], "label_end_s": m["label"]["end_s"],
                                "entity_description": m["label"]["entity_description"],
                                "event_id": m["prediction"].get("event_id"),
                                "entity_id": m["prediction"].get("entity_id"),
                                "pred_start_s": m["prediction"]["start_s"], "pred_end_s": m["prediction"]["end_s"],
                                "start_error_s": m["start_error_s"], "end_error_s": m["end_error_s"]})
        for x in result["missed"]:
            per_behavior[x["behavior"]]["fn"] += 1
            missed_out.append({"clip": entry["name"], "behavior": x["behavior"], "start_s": x["start_s"],
                               "end_s": x["end_s"], "entity_description": x["entity_description"]})
        for p in result["false_positives"]:
            per_behavior[p["behavior"]]["fp"] += 1
            row = {"clip": entry["name"], "behavior": p["behavior"], "start_s": p["start_s"], "end_s": p["end_s"],
                   "event_id": p.get("event_id"), "entity_id": p.get("entity_id")}
            fp_out.append(row)
            if is_none_clip:
                none_details.append(row)
        clips_report[entry["name"]] = {
            "none_clip": is_none_clip, "n_labels": len(real), "n_predictions": len(preds),
            "tp": len(result["matches"]), "fp": len(result["false_positives"]), "fn": len(result["missed"])}

    behavior_report = {}
    for beh, slot in per_behavior.items():
        behavior_report[beh] = {**prf(slot["tp"], slot["fp"], slot["fn"]),
                                "mean_abs_start_error_s": safe_mean(slot["start_errors"]),
                                "mean_abs_end_error_s": safe_mean(slot["end_errors"])}
    all_start = [e for s in per_behavior.values() for e in s["start_errors"]]
    all_end = [e for s in per_behavior.values() for e in s["end_errors"]]
    overall = {**prf(sum(s["tp"] for s in per_behavior.values()), sum(s["fp"] for s in per_behavior.values()),
                     sum(s["fn"] for s in per_behavior.values())),
               "mean_abs_start_error_s": safe_mean(all_start), "mean_abs_end_error_s": safe_mean(all_end)}
    return {
        "tolerance_s": tolerance,
        "clips_evaluated": len(clips_report),
        "clips_skipped_no_events_json": skipped,
        "per_behavior": behavior_report,
        "overall": overall,
        "none_clips": {"n_clips": none_clips,
                       "clips_with_false_alarms": len({d["clip"] for d in none_details}),
                       "false_alarms": len(none_details), "details": none_details},
        "per_clip": clips_report,
        "matches": matches_out,
        "missed": missed_out,
        "false_positives": fp_out,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- text report

def _num(value, digits: int = 2) -> str:
    """Format a ratio / seconds value; None (undefined) prints as '-'."""
    return "-" if value is None else f"{value:.{digits}f}"


def _secs(value) -> str:
    """Seconds with a unit; None (undefined) prints as '-'."""
    return "-" if value is None else f"{value:.2f}s"


def format_report(report: dict, labels_name: str = "labels.csv") -> str:
    """Readable ASCII version of the evaluation report."""
    lines = [f"AquaWatch evaluation   labels: {labels_name}   tolerance: +/- {report['tolerance_s']:g} s   "
             f"clips scored: {report['clips_evaluated']}"]
    if report["clips_skipped_no_events_json"]:
        lines.append("SKIPPED (no events.json found): " + ", ".join(report["clips_skipped_no_events_json"]))
    for w in report["warnings"]:
        lines.append(f"warning: {w}")
    lines += ["", f"{'behaviour':<16}{'TP':>4}{'FP':>4}{'FN':>4}{'precision':>11}{'recall':>8}{'F1':>7}"
                  f"{'|start err|':>13}{'|end err|':>11}",
              "-" * 78]
    used = [(name, r) for name, r in report["per_behavior"].items() if r["tp"] + r["fp"] + r["fn"] > 0]
    unused = [name for name in report["per_behavior"] if name not in {n for n, _ in used}]
    for name, r in used + [("OVERALL", report["overall"])]:
        if name == "OVERALL":
            lines.append("-" * 78)
        lines.append(f"{name:<16}{r['tp']:>4}{r['fp']:>4}{r['fn']:>4}{_num(r['precision']):>11}{_num(r['recall']):>8}"
                     f"{_num(r['f1']):>7}{_secs(r['mean_abs_start_error_s']):>13}{_secs(r['mean_abs_end_error_s']):>11}")
    lines.append("('-' = undefined, nothing to measure. Errors are the mean absolute time difference of matched events.)")
    if unused:
        lines.append("(no labels and no events for: " + ", ".join(unused) + ")")

    nc = report["none_clips"]
    lines += ["", f"Clips labelled 'none': {nc['n_clips']}; false alarms on them: {nc['false_alarms']}"
                  f" (in {nc['clips_with_false_alarms']} clip(s))"]
    for d in nc["details"]:
        lines.append(f"  - {d['clip']}: event {d['event_id']} {d['behavior']} "
                     f"{utils.fmt_time(d['start_s'])}-{utils.fmt_time(d['end_s'])}")

    if report["per_clip"]:
        lines += ["", f"{'clip':<34}{'labels':>7}{'events':>7}{'TP':>4}{'FP':>4}{'FN':>4}", "-" * 60]
        for name, c in report["per_clip"].items():
            tag = " (none)" if c["none_clip"] else ""
            lines.append(f"{(name + tag)[:33]:<34}{c['n_labels']:>7}{c['n_predictions']:>7}"
                         f"{c['tp']:>4}{c['fp']:>4}{c['fn']:>4}")
    if report["matches"]:
        lines += ["", "Matched (TP):"]
        for m in report["matches"]:
            lines.append(f"  - {m['clip']} {m['behavior']}: label {m['label_start_s']:.1f}-{m['label_end_s']:.1f} s, "
                         f"event {m['event_id']} {m['pred_start_s']:.1f}-{m['pred_end_s']:.1f} s "
                         f"(start {m['start_error_s']:+.1f}, end {m['end_error_s']:+.1f} s)")
    if report["missed"]:
        lines += ["", "Missed (FN):"]
        for x in report["missed"]:
            lines.append(f"  - {x['clip']} {x['behavior']} {x['start_s']:.1f}-{x['end_s']:.1f} s  {x['entity_description']}")
    if report["false_positives"]:
        lines += ["", "False alarms (FP):"]
        for p in report["false_positives"]:
            who = "a group" if p["entity_id"] is None else f"#{p['entity_id']}"     # crowding has no single entity
            lines.append(f"  - {p['clip']} event {p['event_id']} {p['behavior']} "
                         f"{p['start_s']:.1f}-{p['end_s']:.1f} s ({who})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- command line

def find_events_file(out_root: Path, key: str) -> Path | None:
    """events.json of one clip inside the outputs folder, or None.

    run.py writes outputs/<clip>/ for the default (campus) scenario and outputs/<clip>_<preset>/ for the other
    presets (pool, pool_hd, pool_trained). The plain folder wins; otherwise the newest preset folder is used.
    """
    candidates = [out_root / key / "events.json"]
    candidates += [out_root / f"{key}_{name}" / "events.json" for name in utils.list_scenarios()]
    found = [path for path in candidates if path.exists()]
    if not found:
        return None
    return found[0] if found[0] == candidates[0] else max(found, key=lambda path: path.stat().st_mtime)


def collect_predictions(labels: list[dict], args) -> dict:
    """Find the events for every labelled clip: from --outputs/<clip_stem>/events.json or from one --events file."""
    keys = {clip_key(r["clip"]) for r in labels}
    if args.events:
        video_in_file, events = load_events(args.events)
        key = clip_key(args.clip or video_in_file or Path(args.events).parent.name)
        if key not in keys:
            raise LabelError(f"no labels for clip '{key}' in {args.labels}; labelled clips: {', '.join(sorted(keys))}")
        return {key: events}
    out_root = Path(args.outputs)
    preds = {}
    for key in sorted(keys):
        path = find_events_file(out_root, key)
        if path is not None:
            preds[key] = load_events(path)[1]
    return preds


def main(argv=None) -> int:
    """CLI entry point. Prints the report and writes eval_report.txt / eval_report.json."""
    for stream in (sys.stdout, sys.stderr):                 # never crash on odd characters in a Windows console
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Score standard events against hand-written labels (precision / recall / F1).")
    ap.add_argument("--labels", required=True, help="labels.csv (clip,entity_description,behavior,start_s,end_s)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--outputs", metavar="DIR", help="folder holding <clip_stem>/events.json for each clip")
    src.add_argument("--events", metavar="events.json", help="score a single events.json")
    ap.add_argument("--clip", help="clip name for --events (default: the video named inside events.json)")
    ap.add_argument("--tolerance", type=float, default=2.0, help="seconds of slack on each side of a label (default 2)")
    ap.add_argument("--report-dir", metavar="DIR", help="where to write eval_report.* (default: next to the labels)")
    args = ap.parse_args(argv)
    if args.tolerance < 0:
        ap.error("--tolerance must be 0 or more")

    try:
        labels = load_labels(args.labels)
        preds = collect_predictions(labels, args)
    except LabelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = evaluate(labels, preds, args.tolerance)
    text = format_report(report, Path(args.labels).name)
    print(text)

    report_dir = utils.ensure_dir(args.report_dir or Path(args.labels).resolve().parent)
    (report_dir / "eval_report.txt").write_text(text + "\n", encoding="utf-8")
    utils.write_json(report_dir / "eval_report.json", report)
    print(f"\nWrote {report_dir / 'eval_report.txt'} and {report_dir / 'eval_report.json'}")
    if report["clips_evaluated"] == 0:
        print("warning: no labelled clip had an events.json, so nothing was scored", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
