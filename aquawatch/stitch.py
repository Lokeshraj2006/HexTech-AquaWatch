"""Join the broken tracks of one person, and leave people on the deck out of the swimmer list. NumPy only.

Why: the tracker gives a NEW ID whenever it loses a person for a moment: under water, behind a splash, or when a
hand-held camera swings. One swimmer then shows up as "7 swimmers", and no single ID lasts long enough for the
5-second distress rule. Steps (all numbers in config.yaml, section `stitch:`):

1. Every raw ID is a TRACKLET (a PIECE): when it starts and ends, its box at both ends, and whether it is a deck
   piece (at least half of its boxes "out_of_water", only with a custom state detector). Not joining a piece is
   always the safe direction: it is exactly what happened before stitching existed.
2. Piece B may CONTINUE piece A when
     - B starts after A ended (they never share a frame: two people seen at the same time are never merged),
     - the gap is at most stitch.max_gap_s,
     - B starts near where A was lost: at most (max_speed_bh_s x gap + slack_bh) body-heights away, counted in the
       SMALLER of the two body sizes (a small far-away swimmer is never joined to someone many of ITS heights away),
     - the box height changed by at most max_size_ratio (the person came closer or the camera zoomed).
3. The cheapest links win first (short gap, short distance, similar size), each piece gets at most one predecessor
   and one successor, and every chain takes the ID of its first piece. Deck pieces are never joined.

Safety: joining must never hide an alert that the separate pieces would have raised. So the result also lists
every join (where a piece ended and the next began) and every piece's time span: stage 3 treats each join like the
end of a track ("lost from view" while in DISTRESS = possible submersion), and stage 2 measures each piece's motion
in that piece's own body size.

drop_deck_tracks() leaves out tracks that were in the water too briefly to ever raise an alert (people on the deck,
false detections on the pool edge). Their boxes stay in the detections, so privacy mode still blurs them.
"""
from __future__ import annotations

import math

import numpy as np

import utils

DECK_STATES = {"out_of_water", "person_out_of_water"}


def deck_classes(cfg: dict) -> set[int]:
    """Class ids of the state detector that mean "on the deck" ({} for the stock COCO detector)."""
    return {k for k, v in utils.state_classes(cfg).items() if v in DECK_STATES}


def _row_class(r) -> int:
    return int(r[8]) if len(r) >= 9 else 0


def _tracklets(rows, deck: set[int]) -> dict[int, dict]:
    """One summary per raw ID: start/end time, end boxes (median of 3), and whether it is a deck piece."""
    by_id: dict[int, list] = {}
    for r in rows:
        if len(r) >= 7:
            by_id.setdefault(int(r[2]), []).append(r)
    out = {}
    for tid, rs in by_id.items():
        rs.sort(key=lambda r: (float(r[1]), int(r[0])))
        boxes = np.asarray([r[3:7] for r in rs], dtype=float)
        n_deck = sum(_row_class(r) in deck for r in rs)
        out[tid] = {"t0": float(rs[0][1]), "t1": float(rs[-1][1]), "n": len(rs),
                    "start": np.median(boxes[:3], axis=0), "end": np.median(boxes[-3:], axis=0),
                    "deck": 2 * n_deck >= len(rs)}                      # at least half of its boxes on the deck
    return out


def _centre(box) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def _height(box) -> float:
    return max(float(box[3] - box[1]), 1.0)


def _body(box) -> float:
    """Body size like features.py (max of height and width): a swimmer turning from flat to upright keeps its size."""
    return max(float(box[3] - box[1]), float(box[2] - box[0]), 1.0)


def link_cost(a: dict, b: dict, scfg: dict) -> float | None:
    """Cost of "piece b continues piece a" (lower = more likely), or None when b cannot be a."""
    gap = b["t0"] - a["t1"]
    if gap <= 0 or gap > float(scfg["max_gap_s"]):
        return None                                        # overlapping in time, or lost for too long
    ha, hb = _height(a["end"]), _height(b["start"])
    # the size test uses the body size (max side), not the height: flat -> upright after a short loss is the same
    # person (bug hunt 2026-10-11, tracking low; PREREGISTRATION_bug_group_C.md step C3)
    ba, bb = _body(a["end"]), _body(b["start"])
    ratio = max(ba, bb) / min(ba, bb)
    max_ratio = float(scfg["max_size_ratio"])
    if ratio > max_ratio:
        return None
    # distance limit in the SMALLER body size: a far (small) victim is never joined to someone far away in its units
    # ...and capped: a 4 s gap must not allow a jump of 5 body sizes onto another object (bug hunt 2026-10-11,
    # tracking high: cam03 #86 -> #111; PREREGISTRATION_bug_group_C.md step C2)
    allowed = min(float(scfg["max_speed_bh_s"]) * gap + float(scfg["slack_bh"]),
                  float(scfg.get("max_jump_bh", float("inf")))) * min(ha, hb)
    dist = float(np.linalg.norm(_centre(a["end"]) - _centre(b["start"])))
    if dist > allowed:
        return None
    return gap / float(scfg["max_gap_s"]) + dist / allowed + math.log(ratio) / math.log(max(max_ratio, 1.0001))


def stitch_tracklets(tracker_result: dict, cfg: dict) -> tuple[dict, dict[int, int]]:
    """Return (tracker_result with joined IDs, {old id: new id} for every ID that was renamed).

    The returned result also holds "stitch_joins" {new id: [[end of a piece, start of the next], ...]} and
    "stitch_pieces" {new id: [[start, end] of every piece], ...} (string keys, JSON-safe) for stages 2 and 3.
    The input is not modified. Disabled (stitch.enabled false) or nothing to join: the same dict and {}.
    """
    scfg = cfg.get("stitch") or {}
    rows = (tracker_result or {}).get("detections") or []
    if not scfg.get("enabled") or len(rows) == 0:
        return tracker_result, {}
    pieces = _tracklets(rows, deck_classes(cfg))
    links = []
    for a_id, a in pieces.items():
        if a["deck"]:
            continue
        for b_id, b in pieces.items():
            if b_id == a_id or b["deck"]:
                continue
            cost = link_cost(a, b, scfg)
            if cost is not None:
                links.append((cost, a["t1"], a_id, b_id))
    links.sort()
    successor: dict[int, int] = {}
    predecessor: dict[int, int] = {}
    for _cost, _t, a_id, b_id in links:                      # cheapest first; one in, one out per piece
        if a_id not in successor and b_id not in predecessor:
            successor[a_id], predecessor[b_id] = b_id, a_id
    rename: dict[int, int] = {}
    joins: dict[str, list] = {}
    spans: dict[str, list] = {}
    for head in sorted(i for i in pieces if i not in predecessor):
        chain = [head]
        while chain[-1] in successor:
            chain.append(successor[chain[-1]])
        if len(chain) < 2:
            continue
        for prev, nxt in zip(chain, chain[1:]):
            rename[nxt] = head
            joins.setdefault(str(head), []).append([pieces[prev]["t1"], pieces[nxt]["t0"]])
        spans[str(head)] = [[pieces[p]["t0"], pieces[p]["t1"]] for p in chain]
    if not rename:
        return tracker_result, {}
    new_rows = [[r[0], r[1], rename.get(int(r[2]), int(r[2])), *r[3:]] for r in rows]
    out = dict(tracker_result)
    out["detections"] = new_rows
    out["stitched"] = {str(k): v for k, v in sorted(rename.items())}
    out["stitch_joins"] = joins
    out["stitch_pieces"] = spans
    return out, rename


def drop_deck_tracks(tracks: dict, tracker_result: dict, cfg: dict) -> tuple[dict, list[int]]:
    """Leave out tracks that were in the water too briefly to ever raise an alert: people on the deck and false
    detections on the pool edge. Only with a custom state detector; returns (swimmer tracks, dropped IDs).

    A track is left out when its boxes that are NOT "out_of_water" span less than half of pool.hold_s. A distress
    alert needs the risk held for hold_s on in-water samples, so such a track can never alert: leaving it out cannot
    hide an alert. Someone who sits on the edge and then slips in and struggles stays in (their in-water part is long).
    """
    deck = deck_classes(cfg)
    if not deck:
        return tracks, []
    limit = 0.5 * float(cfg["pool"]["hold_s"])
    first: dict[int, float] = {}
    last: dict[int, float] = {}
    for r in (tracker_result or {}).get("detections") or []:
        if len(r) >= 7 and _row_class(r) not in deck:
            tid, t = int(r[2]), float(r[1])
            first[tid] = min(first.get(tid, t), t)
            last[tid] = max(last.get(tid, t), t)
    dropped = [tid for tid in tracks if last.get(tid, -1.0) - first.get(tid, 0.0) < limit]
    return {tid: tr for tid, tr in tracks.items() if tid not in dropped}, dropped


def track_gaps(tracks: dict, min_gap_s: float, tracker_result: dict | None = None) -> dict[int, list[tuple]]:
    """{track id: [(last time seen, time seen again), ...]} for stage 3: every gap of at least min_gap_s inside a
    track, plus EVERY join made by stitch_tracklets whatever its length (where one piece ended is where the old,
    unjoined track ended: stage 3 must judge it exactly as before)."""
    out: dict[int, set] = {}
    # times from ALL boxes of each ID (also the ones too small to keep in the track): a hole made by the size filter
    # is not 'lost from view' (bug hunt 2026-10-11, tracking low; PREREGISTRATION_bug_group_C.md step C4)
    seen: dict[int, list] = {}
    for row in (tracker_result or {}).get("detections") or []:
        if len(row) >= 3 and int(row[2]) in tracks:
            seen.setdefault(int(row[2]), []).append(float(row[1]))
    for tid, tr in tracks.items():
        t = np.unique(np.asarray(seen[int(tid)], dtype=float)) if int(tid) in seen else np.asarray(tr.t, dtype=float)
        t = t[(t >= float(tr.t[0]) - 1e-9) & (t <= float(tr.t[-1]) + 1e-9)] if len(tr.t) else t
        for i in range(len(t) - 1):
            if t[i + 1] - t[i] >= min_gap_s:
                out.setdefault(int(tid), set()).add((float(t[i]), float(t[i + 1])))
    for tid, joins in ((tracker_result or {}).get("stitch_joins") or {}).items():
        if int(tid) in tracks:
            for end, start in joins:
                out.setdefault(int(tid), set()).add((float(end), float(start)))
    return {tid: sorted(g) for tid, g in out.items()}


def piece_spans(tracker_result: dict | None) -> dict[int, list[tuple[float, float]]]:
    """{track id: [(start, end) of each joined piece]} (empty for tracks that were not joined)."""
    raw = (tracker_result or {}).get("stitch_pieces") or {}
    return {int(k): [(float(a), float(b)) for a, b in v] for k, v in raw.items()}
