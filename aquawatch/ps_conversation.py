"""AquaWatch "Ask the Video": grounded answers about a multi-camera pool site.

    python ps_conversation.py --db outputs/virtual_cctv/aquawatch.sqlite3 "what happened on cam02?"

Every answer is built only from rows the real pipeline wrote into the event index (event_index.py: events, tracks,
cameras, cross-camera links, identity candidates) and from places the USER defined in chat (memory.py). Nothing is
guessed:
  * a tracker ID is local to one camera ("Track #3 (cam02)"), never a person;
  * a cross-camera link is quoted with its own status: "verified_automated (known camera geometry)" or
    "Candidate match - unverified (appearance score 0.83)"; without either: "No cross-camera identity evidence.";
  * a place the index does not know ("the deep end") is asked about ONCE, then remembered across restarts;
  * visual descriptions ("a child in a blue life jacket", "a big splash") go to the open-vocabulary CLIP search
    (openvocab.py) when the site has an openvocab_index next to the event index: every hit is a camera + time + track
    + evidence image, labelled as similarity (not verification), and a weak best match gives "no confident match".
    Without that index they are refused with CAPABILITY_REFUSAL instead of guessed. Faces, race, skin colour, gender
    and age are always refused (privacy: no identification or demographic guessing);
  * nothing found -> status "no_evidence" (which is not proof that nothing happened).
The single-camera ConversationIndex of the team repo is replaced by this site-wide engine (same idea, many cameras).
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cameras as camreg
import identity
import memory
from event_index import EventIndex

CAPABILITY_REFUSAL = "I cannot verify that visual attribute with the currently enabled vision models."
NO_ALIGNMENT = "Temporal alignment unavailable."
BEHAVIOUR_TEXT = {"aquatic_warning": "Level 1 early warning (swimmer may need help)",
                  "aquatic_distress": "Level 2 distress alarm", "submersion": "Level 3 possible submersion"}
UNDER_WATER_TEXT = "URGENT possibly under water (out of sight, then seen again at the same spot)"


def behaviour_text(event: dict) -> str:
    """How an indexed alert is named: its level, or the urgent 'possibly under water' for a Rule C alert."""
    if "back_at_same_spot" in ((event.get("details") or {}).get("evidence_keys") or []):
        return UNDER_WATER_TEXT
    return BEHAVIOUR_TEXT.get(event.get("behavior"), str(event.get("behavior")))
HELP = ("I can answer from the indexed footage: what happened (overall, on a camera, at a saved place, between two "
        "times); which camera saw the highest risk; how many tracks were in view on a camera at a time; the evidence "
        "for an event or for 'track 3 on cam04'; whether a track was also seen on another camera; and which cameras "
        "exist and how their clocks are aligned; and, when the site has an open-vocabulary index, find things you "
        "describe ('is there a child in a blue life jacket?'). Teach me places with e.g. 'CAM03 is the deep end'.")
AT_WINDOW_S = 2.0                                # "at 15:00:12" means +-2 s

# visual attributes no enabled model measures (detector = person boxes, behaviour = posture/motion, Re-ID = colour
# histogram used only for ranking candidates): never answered from the footage
_ATTRIBUTE = re.compile(
    r"\b(colou?rs?|red|blue|green|yellow|black|white|orange|purple|pink|brown|grey|gray|cap|caps|hat|swimsuit|"
    r"swimwear|costume|trunks|bikini|shirt|wearing|wears|wore|dressed|clothes|clothing|age|aged|old|young|child|"
    r"children|kid|kids|boy|girl|man|men|woman|women|male|female|gender|face(?![\s-]+(?:down|up|first)\b)|faces|"
    r"hair|tattoo|beard|glasses|"
    r"goggles|tall|skin|race|ethnicity)\b")
# never searched, even with the open-vocabulary index: identification / demographic guesses (privacy).
# Not refused: "floating face down / face up" (a posture, a drowning sign), "identify the swimmer in distress" and
# "who is this swimmer in trouble?" (pointing someone out, not naming them).
_SENSITIVE = re.compile(r"\b(faces?(?![\s-]+(?:down|up|first)\b)|facial|race|racial|ethnicity|ethnic|"
                        r"skin(?: colou?r)?|gender|male|female|how old|age|aged|tattoos?|identity of|name of|"
                        r"identify (?:him|her|them|this person|that person)|recogni[sz]e|who is (?:he|she)|"
                        r"(?:a )?man or (?:a )?woman|(?:a )?woman or (?:a )?man|(?:a )?boy or (?:a )?girl|"
                        r"(?:a )?girl or (?:a )?boy)\b"
                        r"|\bwho is (?:this|that)(?: (?:person|man|woman|guy|lady|girl|boy|kid|child))?\s*\??\s*$")
# a description by colour or clothing is always a picture search (never routed elsewhere)
_HARD_VISUAL = re.compile(r"\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|grey|gray|cap|caps|hat|"
                          r"swimsuit|swimwear|costume|trunks|bikini|shirt|t-shirt|wearing|wears|goggles|towel|"
                          r"umbrella|inflatable|float|tube|ring|holding|carrying)\b")
PRIVACY_REFUSAL = ("I don't search for faces, race, skin colour, gender or age: AquaWatch never identifies people or "
                   "guesses demographics. Describe what is visible instead (clothing colour, floats, actions).")
# a request to find something visible ("is there a ...", "find a ...", "did a ... pass")
_VISUAL_CMD = re.compile(r"^(?:(?:please|can you|could you|ok(?:ay)?|hey)[, ]+)*(?:find|search|look for|locate|"
                         r"show(?: me)? (?:a|an|any|the|someone|somebody|people|persons?)\b|where (?:is|was|are|were)|"
                         r"when (?:is|was|did)|is there|are there|was there|were there|did (?:a|an|any|anyone|someone|"
                         r"somebody|the)\b|did you see|do you see|have you seen|any(?:one|body)? (?:with|in|wearing|"
                         r"holding|carrying|on)\b|which cameras? (?:saw|shows?|showed|has|had) (?:a|an|any|the|someone)\b)")
VISUAL_SHOWN = 3                                   # hits written out in the answer (all are cited)
DETECTOR_NAMES = {"yoloworld": "YOLO-World", "yoloe": "YOLOE"}
# 'track 15', and the words the app's own answers use: 'Swimmer #15', 'ID 15', 'person 15'
_TRACK = re.compile(r"\b(?:track|swimmer|id|person)\s*#?\s*(\d+)\b")
_CAMERA = re.compile(r"\b" + memory.CAMERA_TOKEN + r"\b")
_WALL = r"(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)(?:\s*(am|pm)\b)?"
_VIDEO = r"(?:(\d{1,2}):(\d{2}(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b)"
# the first number of a pair may leave the unit to the second one: 'between 5 and 10 seconds'
_VIDEO_FIRST = r"(?:(\d{1,2}):(\d{2}(?:\.\d+)?)|(\d+(?:\.\d+)?)(?:\s*(?:s|sec|secs|seconds?)\b)?)"
# 'the last 10 minutes', 'first 5 seconds', 'the final minute', 'past 2 hours' (relative to the footage)
_RELATIVE = re.compile(r"\b(?P<side>last|final|past|first|opening)\s+(?:(?P<n>\d+(?:\.\d+)?)\s*"
                       r"(?P<unit>s|sec|secs|seconds?|min|mins|minutes?|h|hrs?|hours?)|(?P<bare>second|minute|hour))\b")
# something that looks like a time; when no window could be read from it, the answer says so
_TIME_HINT = re.compile(r"\d+(?:\.\d+)?\s*(?:s|sec|secs|seconds?|min|mins|minutes?|h|hours?)\b|\b\d{1,2}:\d{2}\b")
_IDENTITY = re.compile(r"\b(same|also|appear\w*|other cameras?|another camera|re-?id|identity|identities|match\w*|"
                       r"linked|link|global)\b")
_CAMERAS = re.compile(r"\b(which cameras|list (?:the )?cameras|cameras? (?:are there|exist)|sync\w*|alignment|aligned|"
                      r"offsets?|(?<!o')(?<!o\u2019)clocks?)\b")
_RISK = re.compile(r"\b(highest|most|max(?:imum)?|worst|peak|top)\b.*\brisk\b|\brisk\b.*\b(highest|most|max)\b|"
                   r"most (?:dangerous|urgent)")
_EVENTS = re.compile(r"what happened|\b(events?|alerts?|alarms?|warnings?|distress|submersion|submerged|went under|"
                     r"incidents?|anything|summary|summari[sz]e|drown\w*|happen\w*|going on)\b")
_EVIDENCE = re.compile(r"\b(evidence|snapshots?|proof|show|picture|image|why)\b")
# "who is drowning / the id of the swimmer in trouble": answered with flagged tracks, or the closest ones if none
_WHO = re.compile(r"\b(drown\w*|distress\w*|struggl\w*|in trouble|needs? help|victim|in danger|at risk)\b")
_WHO_TARGET = re.compile(r"\b(id|ids|which|who|track|tracks|swimmer|swimmers|person|people|one)\b")
CLOSEST_TRACKS = 3                               # tracks listed when nobody was flagged
CLIP_BEFORE_S, CLIP_AFTER_S = 3.0, 5.0           # a chat answer's clip: this long before / after the cited moment


def _wall_text(moment: datetime) -> str:
    """Index wall-clock text 'YYYY-MM-DDTHH:MM:SS.mmm' (compares as a string)."""
    return camreg.format_wall(moment)


def _clock(wall: str | None) -> str:
    """'2026-10-09T15:00:14.100' -> '15:00:14.1' for answers."""
    if not wall:
        return "?"
    text = str(wall)[11:23].rstrip("0").rstrip(".")
    return text


def _vtime(seconds: Any) -> str:
    """Video seconds -> 'mm:ss.s'."""
    s = round(float(seconds or 0.0), 1)              # round first: 59.97 s is 01:00.0, not 00:60.0
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


class SiteConversation:
    """Question answering over one site's event index plus the user's saved references."""

    def __init__(self, db_path: str | Path, references_path: str | Path | None = None):
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(f"no event index at {self.db_path} (run ingest.py first)")
        self.index = EventIndex(self.db_path)
        self.memory = memory.ReferenceMemory(references_path or self.db_path.with_name("references.sqlite3"))
        self.cameras = {c["camera_id"]: c for c in self.index.list_cameras()}
        self.dataset_label = self.index.get_meta("dataset_label")
        self.pending: str | None = None          # the question waiting for a place definition
        self.pending_place: str | None = None    # ...and the place it asked about (a bare camera reply defines it)
        self._visual_index: dict | None = None   # openvocab index, loaded on the first visual question

    def visual_index(self) -> dict | None:
        """The site's open-vocabulary index (openvocab_index.npz next to the event index), or None."""
        if self._visual_index is None:
            import openvocab
            self._visual_index = openvocab.load_index(self.db_path.parent) or {}
        return self._visual_index or None

    # ------------------------------------------------------------------ helpers

    def camera_name(self, camera_id: str) -> str:
        """Registry name of a camera ('CAM02 Centre pool'), else the id upper-cased."""
        cam = self.cameras.get(camera_id) or {}
        return cam.get("name") or str(camera_id).upper()

    def single(self) -> bool:
        """One camera only (the single-video view): answers then talk about 'Swimmer #2 at 00:03.5', no camera ids."""
        return len(self.cameras) == 1

    def who(self, track, camera_id: str) -> str:
        """'Swimmer #2' for a single video, 'Track #2 (cam01)' across cameras (a tracker id is local to one camera)."""
        return f"Swimmer #{int(track)}" if self.single() else identity.display_track(track, camera_id)

    def when(self, wall, camera_id: str, video_s) -> str:
        """'00:03.5' for a single video (its clock IS the video time), else '15:00:03.5 (cam01 video 00:03.5)'."""
        return _vtime(video_s) if self.single() else f"{_clock(wall)} ({camera_id} video {_vtime(video_s)})"

    def _result(self, answer: str, status: str, citations: list | None = None, clarify: dict | None = None,
                used: list | None = None) -> dict:
        """The answer dict every question returns. Every cited moment gets a "clip" when its camera has an annotated
        video (boxes drawn, faces blurred in privacy mode): {"video", "start_s", "end_s"} around the cited time."""
        for c in citations or []:
            if "clip" not in c:
                c["clip"] = self._clip(c)
        return {"answer": answer, "status": status, "citations": citations or [], "clarify": clarify,
                "references_used": used or [], "dataset_label": self.dataset_label}

    def _clip(self, citation: dict) -> dict | None:
        """The annotated video of the cited camera from CLIP_BEFORE_S before to CLIP_AFTER_S after the cited time
        (never the original video: that would show the faces privacy mode blurred). None when there is none."""
        cam = self.cameras.get(citation.get("camera_id")) or {}
        t = citation.get("time_s")
        if t is None or not cam.get("out_dir"):
            return None
        video = camreg.resolve_path(str(cam["out_dir"])) / "annotated.mp4"
        if not video.is_file():
            return None
        return {"video": str(video), "start_s": round(max(0.0, float(t) - CLIP_BEFORE_S), 2),
                "end_s": round(float(t) + CLIP_AFTER_S, 2)}

    def _citation(self, event: dict) -> dict:
        """Citation of one indexed event: source, camera, time, track, state, risk, evidence, verification status."""
        cam, track = event["camera_id"], event.get("person_id")
        gid = self._global_id(cam, track) if track is not None else None
        return {"camera_id": cam, "camera_name": self.camera_name(cam), "source": event.get("video_path"),
                "track_id": track, "event": event.get("behavior"),
                "event_state": behaviour_text(event),
                "time_s": event.get("alert_s") if event.get("alert_s") is not None else event.get("start_s"),
                "wall_time": event.get("alert_wall") or event.get("start_wall"), "snapshot": event.get("snapshot"),
                "confidence": event.get("confidence"), "evidence": event.get("evidence"),
                "verification": (f"{identity.VERIFIED_AUTOMATED} (known camera geometry, global id {gid})" if gid
                                 else "local track only: no verified cross-camera identity")}

    def _geometry_partners(self, camera_id: str, track_id: Any) -> list[dict]:
        """Verified links of one local track: [{camera, track, method, status, evidence}] (other side only). A track
        without a global id has no verified partner, even when an older index still stores a link as verified (links in a
        contradictory group, bug hunt 2026-10-11)."""
        if track_id is None or self._global_id(camera_id, track_id) is None:
            return []
        out = []
        for link in self.index.links(camera_id=camera_id, track_id=int(track_id)):
            if link["status"] not in identity.VERIFIED_STATUSES:
                continue
            mine = link["camera_a"] == camera_id and int(link["track_a"]) == int(track_id)
            out.append({"camera": link["camera_b"] if mine else link["camera_a"],
                        "track": link["track_b"] if mine else link["track_a"], "method": link["method"],
                        "status": link["status"], "evidence": link.get("evidence") or {}})
        return out

    def _global_id(self, camera_id: str, track_id: Any) -> str | None:
        """Global id given by the geometry links (meta), or None (no evidence -> no global identity)."""
        groups = self.index.get_meta("geometry_global_ids") or {}
        for gid, members in groups.items():
            if any(m[0] == camera_id and int(m[1]) == int(track_id) for m in members):
                return gid
        return None

    def _event_line(self, e: dict) -> str:
        """One plain line about an event, with its cross-camera partners when geometry links exist."""
        cam = e["camera_id"]
        what = behaviour_text(e)
        risk = f", risk {float(e['confidence']):.0%}" if e.get("confidence") is not None else ""
        alert_s = e.get("alert_s") if e.get("alert_s") is not None else e.get("start_s")
        if self.single():
            line = (f"{self.who(e.get('person_id'), cam)}: {what} at {_vtime(alert_s)}{risk}, warning signs from "
                    f"{float(e['start_s']):.1f} to {float(e['end_s']):.1f} s")
        else:
            line = (f"{_clock(e.get('alert_wall') or e.get('start_wall'))} ({cam} video {_vtime(alert_s)}) "
                    f"{self.camera_name(cam)}: {what} for {identity.display_track(e.get('person_id'), cam)}{risk}, "
                    f"behaviour {float(e['start_s']):.1f}-{float(e['end_s']):.1f} s")
        partners = self._geometry_partners(cam, e.get("person_id"))
        if partners:
            line += ("; same swimmer as " + ", ".join(identity.display_track(p["track"], p["camera"]) for p in partners)
                     + " (verified_automated: known camera geometry)")
        if len(self.cameras) > 1 and (self.cameras.get(cam) or {}).get(
                "sync_status", camreg.SYNC_UNAVAILABLE) == camreg.SYNC_UNAVAILABLE:
            line += f". {NO_ALIGNMENT}"                  # alignment only matters between cameras
        return line

    def _window(self, q: str) -> dict | None:
        """Time window in the question: wall clock (hh:mm:ss) or video seconds ('12 s', '00:12'); None if none."""
        date = None
        if self.cameras:
            date = str(next(iter(self.cameras.values())).get("start_time") or "")[:10] or None

        def wall(match_groups) -> datetime | None:
            if date is None:
                return None
            h, m, s = int(match_groups[0]), int(match_groups[1]), float(match_groups[2])
            half = match_groups[3] if len(match_groups) > 3 else None
            if half == "pm" and h < 12:
                h += 12
            elif half == "am" and h == 12:
                h = 0
            return datetime.fromisoformat(date) + timedelta(hours=h, minutes=m, seconds=s)

        def video(groups) -> float:
            return int(groups[0]) * 60 + float(groups[1]) if groups[0] is not None else float(groups[2])

        longest = max((float(c.get("duration_s") or 0) for c in self.cameras.values()), default=0.0)

        def clock_instead(groups, t: float) -> datetime | None:
            """'15:00' read as mm:ss is 900 s, longer than every video: if it falls on the cameras' clock, it is the
            clock time 15:00:00 (bug hunt 2026-10-11)."""
            if groups[0] is None or not longest or t <= longest + AT_WINDOW_S:
                return None
            moment = wall((groups[0], groups[1].split(".")[0], "0"))
            if moment is None:
                return None
            starts = [datetime.fromisoformat(str(c.get("start_time"))) for c in self.cameras.values()
                      if c.get("start_time")]
            if starts and min(starts) - timedelta(minutes=1) <= moment <= max(starts) + timedelta(seconds=longest + 60):
                return moment
            return None

        rel = _RELATIVE.search(q)
        if rel:                                           # "the last 10 minutes", "the first 5 seconds"
            return self._relative_window(rel, longest)

        for pattern, kind in ((_WALL, "wall"), (_VIDEO, "video")):
            first = _VIDEO_FIRST if kind == "video" else pattern
            pair = re.search(rf"\b(?:between|from)\s+{first}\s+(?:and|to|-)\s+{pattern}", q)
            n = 4 if kind == "wall" else 3
            if pair:
                g = pair.groups()
                a, b = (wall(g[:n]), wall(g[n:])) if kind == "wall" else (video(g[:n]), video(g[n:]))
                if a is not None and b is not None:
                    return {"kind": kind, "start": min(a, b), "end": max(a, b), "text": pair.group(0)}
            for word, side in (("after|since", "after"), ("before|until", "before"), ("at|around|near", "at")):
                one = re.search(rf"\b(?:{word})\s+{pattern}", q)
                if not one:
                    continue
                t = wall(one.groups()) if kind == "wall" else video(one.groups())
                if t is None:
                    continue
                if kind == "video" and clock_instead(one.groups(), t) is not None:
                    t, kind = clock_instead(one.groups(), t), "wall"    # hh:mm on the cameras' clock
                pad = timedelta(seconds=AT_WINDOW_S) if kind == "wall" else AT_WINDOW_S
                start = t - pad if side == "at" else (t if side == "after" else None)
                end = t + pad if side == "at" else (t if side == "before" else None)
                if kind == "video" and start is not None:
                    start = max(0.0, start)
                return {"kind": kind, "start": start, "end": end, "text": one.group(0)}
        return None

    def _relative_window(self, rel: re.Match, longest: float) -> dict:
        """'last N units' / 'first N units' counted from the footage's own end / start (there is no live clock):
        a clock window when the cameras have start times, else video seconds. Says so when the footage is shorter."""
        number = float(rel.group("n")) if rel.group("n") else 1.0
        unit = (rel.group("unit") or rel.group("bare") or "s").lower()
        seconds = number * (3600.0 if unit.startswith("h") else 60.0 if unit.startswith("m") else 1.0)
        last = rel.group("side").lower() in ("last", "final", "past")
        starts, ends = [], []
        for cam in self.cameras.values():
            try:
                start = datetime.fromisoformat(str(cam.get("start_time")))
            except (TypeError, ValueError):
                continue
            starts.append(start)
            ends.append(start + timedelta(seconds=float(cam.get("duration_s") or 0)))
        span = (max(ends) - min(starts)).total_seconds() if starts else longest
        note = (f"the footage is only {span:.1f} s long, so this covers all of it" if span and seconds >= span
                else f"counted from the {'end' if last else 'start'} of the footage, not from now")
        if starts:
            w = ({"kind": "wall", "start": max(ends) - timedelta(seconds=seconds), "end": max(ends)} if last else
                 {"kind": "wall", "start": min(starts), "end": min(starts) + timedelta(seconds=seconds)})
        else:
            w = ({"kind": "video", "start": max(0.0, longest - seconds), "end": longest} if last else
                 {"kind": "video", "start": 0.0, "end": seconds})
        return dict(w, text=rel.group(0), note=note)

    def _window_text(self, w: dict | None) -> str:
        """' between 15:00:05 and 15:00:15' / ' in video time 10.0-15.0 s' / '' (+ the window's note, if any)."""
        if not w:
            return ""
        if w.get("note"):
            return self._window_text({k: v for k, v in w.items() if k != "note"}) + f" ({w['note']})"
        if w["kind"] == "wall":
            a = _clock(_wall_text(w["start"])) if w["start"] else "the start"
            b = _clock(_wall_text(w["end"])) if w["end"] else "the end"
            return f" between {a} and {b}"
        a = f"{w['start']:.1f}" if w["start"] is not None else "0"
        b = f"{w['end']:.1f}" if w["end"] is not None else "the end"
        return f" in video time {a}-{b} s"

    def _events(self, cams: list[str] | None, w: dict | None, behavior: str | None = None,
                track: int | None = None) -> list[dict]:
        """Indexed events filtered by cameras, window, behaviour and track (chronological)."""
        rows = []
        for cam in cams or [None]:
            if w and w["kind"] == "wall":
                rows += self.index.events_between(
                    _wall_text(w["start"]) if w["start"] else None, _wall_text(w["end"]) if w["end"] else None,
                    camera_id=cam, behavior=behavior, person_id=track)
            else:
                found = self.index.query(camera_id=cam, behavior=behavior, person_id=track)
                if w:                                    # video seconds: overlap with the window
                    found = [e for e in found if (w["start"] is None or float(e["end_s"]) >= w["start"])
                             and (w["end"] is None or float(e["start_s"]) <= w["end"])]
                rows += found
        return sorted(rows, key=lambda e: (e.get("alert_wall") or e.get("start_wall") or "", e["camera_id"]))

    def _scope(self, q: str) -> tuple[list[str] | None, list[dict], dict | None]:
        """Cameras named in the question (ids or saved places) -> (camera ids or None, references used, clarify)."""
        cams: list[str] = []
        for token in _CAMERA.findall(q):
            cam = memory.resolve_camera_token(token, self.cameras)
            if cam and cam not in cams:
                cams.append(cam)
        used = []
        for ref in self.memory.find_in_text(q):
            used.append(ref)
            if ref["camera_id"] not in cams:
                cams.append(ref["camera_id"])
        if not used:
            place = memory.find_place_phrase(q)
            if place:
                named = [c for c, row in self.cameras.items() if place in str(row.get("name") or "").lower()]
                by_core = self.memory.find_by_core(place)      # 'deep part' = the saved 'deep end' (same core)
                if len(named) == 1:                         # the camera's own registry name ("left wall")
                    cams.append(named[0])
                elif len(by_core) == 1:
                    used.append(dict(by_core[0], matched_from=place))
                    if by_core[0]["camera_id"] not in cams:
                        cams.append(by_core[0]["camera_id"])
                elif len(by_core) > 1:                      # two saved places fit: ask, never guess
                    options = " or ".join(f"the {r['label']} ({str(r['camera_id']).upper()})" for r in by_core)
                    question = (f"By '{place}', do you mean {options}? Please use the name you saved.")
                    return None, [], {"reference": memory.normalize_reference(place), "question": question,
                                      "kind": "ambiguous_place"}
                elif not any(r["reference"] == memory.normalize_reference(place) for r in self.memory.all()):
                    known = ", ".join(c.upper() for c in self.cameras)
                    question = (f"Which camera covers the {place}? Nothing in the footage says so. Tell me once, "
                                f"e.g. 'CAMxx is the {place}' or just the camera (cameras: {known}), and I will "
                                f"remember it.")
                    return None, [], {"reference": memory.normalize_reference(place), "question": question,
                                      "kind": "unknown_place"}
        return (cams or None), used, None

    # ------------------------------------------------------------------ answering

    def answer(self, question: str, visual_method: str | None = None) -> dict:
        """Answer one question or statement (see the module docstring for the rules). `visual_method` overrides the
        open-vocabulary method for this one question (e.g. "ground_yoloe" for the chat's "Check with YOLOE").
        The answer type chosen is kept in self.last_intent (tools/eval_chat.py measures it)."""
        self._visual_method = visual_method
        self.last_intent = None
        raw = str(question or "").strip()
        q = re.sub(r"\s+", " ", raw.lower())
        if not q:
            return self._result(HELP, "unsupported")
        statement = memory.parse_statement(raw, self.cameras)
        # a bare camera ("cam03") answers "which camera covers the deep end?" - only as the very next message (audit F01)
        waiting_place, self.pending_place = self.pending_place, None
        reply = memory.camera_reply(raw, self.cameras) if (self.pending and waiting_place and not statement) else None
        if reply:
            token, cam = reply
            if not cam:                              # "cam09": not a known camera, so still waiting for the answer
                self.pending_place = waiting_place
            statement = {"reference": waiting_place, "label": memory.reference_label(waiting_place),
                         "camera": cam, "camera_token": token, "region": None}
        if statement:
            self.last_intent = "statement"
            return self._remember(statement, raw)
        forget = re.match(r"^(?:please\s+)?forget\s+(?:about\s+)?(?:the\s+)?(.+?)[.!]?$", q)
        if forget:
            self.last_intent = "forget"
            ok = self.memory.forget(forget.group(1), provenance=f"chat: '{raw}'")
            return self._result(f"Forgot the saved reference '{forget.group(1)}' (kept in its history)." if ok else
                                f"I have no saved reference called '{forget.group(1)}'.", "answered")
        if _SENSITIVE.search(q):
            self.last_intent = "privacy"
            return self._result(PRIVACY_REFUSAL, "unsupported")
        track_m = _TRACK.search(q)
        track = int(track_m.group(1)) if track_m else None
        # 1. hard rules: an explicit track number, a colour / clothing description, the camera list (exact words);
        # 2. otherwise the meaning router when it is clearly sure; 3. otherwise the word rules.
        if track is not None:
            intent = self.rule_intent(q, track)
        elif _HARD_VISUAL.search(q):
            intent = "visual"
        elif _CAMERAS.search(q) and not self.single():
            intent = "cameras"
        else:
            intent = self.semantic_intent(q) or self.rule_intent(q, track)
        self.last_intent = intent or "unsupported"
        if intent == "visual" and self.visual_index() is None:
            return self._result(CAPABILITY_REFUSAL, "unsupported")
        if intent == "help":
            return self._result(HELP, "answered")
        if intent == "cameras":
            return self._cameras_answer()
        cams, used, clarify = self._scope(q)
        if clarify:
            self.pending = raw
            self.pending_place = clarify.get("reference") if clarify.get("kind") == "unknown_place" else None
            return self._result(clarify["question"], "needs_clarification", clarify=clarify)
        note = "".join(f" (using your saved reference: {memory.describe(r)}"
                       + (f"; I read '{r['matched_from']}' as '{r['label']}'" if r.get("matched_from") else "") + ")"
                       for r in used)
        w = self._window(q)
        if w is None and _TIME_HINT.search(q):           # never drop a time silently (bug hunt 2026-10-11)
            note += " (I could not read the time in your question, so this covers the whole video)"
        if intent == "visual":
            return self._visual_answer(raw, cams, w, note, used)
        if track is not None and not cams:
            have = sorted(c for c in self.cameras if self.index.query_tracks(camera_id=c, track_id=track))
            if len(have) != 1:
                where = ", ".join(have) if have else "no camera"
                return self._result(f"Track numbers are local to each camera; Track #{track} exists on {where}. "
                                    f"Which camera do you mean (e.g. 'track {track} on {(have or ['cam01'])[0]}')?",
                                    "needs_clarification" if have else "no_evidence")
            cams = have
        if intent is None and (w or cams):                   # only a time window or a place: what happened there
            intent = self.last_intent = "events"
        if intent == "identity":
            return self._identity_answer(cams, track, note, used)
        if intent == "risk":
            return self._risk_answer(cams, w, note, used)
        if intent == "who":
            return self._who_answer(cams, w, note, used)
        if intent == "count":
            return self._count_answer(cams, w, note, used)
        if intent == "evidence":
            return self._evidence_answer(cams, w, track, note, used)
        if intent == "events":
            behavior = ("submersion" if re.search(r"submer|went under|under water", q) else
                        "aquatic_distress" if re.search(r"\bdistress\b|\balarm", q) else
                        "aquatic_warning" if re.search(r"\bwarning", q) else None)
            return self._events_answer(cams, w, behavior, note, used)
        return self._result("I can't answer that from the indexed footage. " + HELP, "unsupported")

    def rule_intent(self, q: str, track: int | None) -> str | None:
        """The answer type the word rules recognise in a (lower-cased) question, or None. Same order as always:
        visual description, help, cameras, identity, risk, who is in trouble, how many, evidence, events."""
        visual = track is None and bool(_ATTRIBUTE.search(q) or (_VISUAL_CMD.search(q) and not (
            _EVENTS.search(q) or _RISK.search(q) or _CAMERAS.search(q) or re.search(r"\bhow many\b", q)
            or re.search(r"\b(evidence|proof|snapshots?)\b", q)      # "show the evidence": an alert's evidence
            or (_WHO.search(q) and _WHO_TARGET.search(q)))))
        if visual:
            return "visual"
        if re.search(r"^help\b|\bwhat can you\b|\bwhat do you know\b|\bhow (?:do|can) i use\b", q):
            return "help"                        # not "needs help": that is someone in trouble
        if _CAMERAS.search(q):
            return "cameras"
        if _IDENTITY.search(q) and not _RISK.search(q):
            return "identity"
        if _RISK.search(q):
            return "risk"
        if _WHO.search(q) and _WHO_TARGET.search(q) and track is None:
            return "who"
        if re.search(r"\bhow many\b", q):
            return "count"
        if track is not None or _EVIDENCE.search(q):
            return "evidence"
        if _EVENTS.search(q):
            return "events"
        return None

    def semantic_intent(self, q: str) -> str | None:
        """The meaning router (chat_router.py), asked before the word rules; None when it is not clearly sure (then
        the word rules decide) or cannot run (no CLIP weights)."""
        try:
            import chat_router
            return chat_router.intent(q, multi_camera=not self.single())
        except Exception:                                    # no router / no model: the rules' answer stands
            return None

    def _remember(self, statement: dict, raw: str) -> dict:
        """Store a user-defined place, then answer the question that was waiting for it (if any)."""
        if not statement.get("camera"):
            known = ", ".join(c.upper() for c in self.cameras)
            return self._result(f"I don't know a camera called '{statement['camera_token']}'. Cameras: {known}.",
                                "needs_clarification")
        record = self.memory.remember(statement["reference"], statement["camera"], statement.get("region"),
                                      provenance=f"chat: '{raw}'")
        text = f"Saved: {memory.describe(record)}. Version {record['version']}."
        same = [r for r in self.memory.find_by_core(record["label"]) if r["reference"] != record["reference"]]
        if same:                                     # say it: questions about this meaning will now ask which one
            text += (" Note: " + ", ".join(f"'{r['label']}' ({str(r['camera_id']).upper()})" for r in same)
                     + " means the same; say 'forget the " + same[0]["label"] + "' if this one replaces it.")
        pending, self.pending, self.pending_place = self.pending, None, None
        if pending:
            follow = self.answer(pending)
            follow["answer"] = f"{text}\n\nYour earlier question - {pending}\n{follow['answer']}"
            return follow
        return self._result(text, "answered", used=[record])

    def _video_window(self, w: dict | None, cams: list[str]) -> dict | None:
        """The question's window as video seconds per camera ({camera: (start, end)}), or None."""
        if not w:
            return None
        if w["kind"] == "video":
            return {c: (w["start"], w["end"]) for c in cams}
        out = {}
        for c in cams:
            try:
                start = datetime.fromisoformat(str(self.cameras[c].get("start_time")))
            except (TypeError, ValueError):
                continue
            # clock time = start_time + video seconds (cameras.wall_clock, event_index): offset_s is NOT part of it;
            # it only places a camera on the shared site timeline (bug hunt 2026-10-11, multicam low)
            out[c] = tuple(None if t is None else (t - start).total_seconds() for t in (w["start"], w["end"]))
        return out

    def _visual_answer(self, raw: str, cams, w, note, used) -> dict:
        """Open-vocabulary search (openvocab.py): best matches with camera, time, track and an evidence image."""
        import openvocab
        settings = openvocab.load_settings()
        drop = [m.group(0) for m in _CAMERA.finditer(raw.lower())] + ([w["text"]] if w else [])
        drop += [f"the {r['label']}" for r in used] + [str(r["label"]) for r in used]
        text = openvocab.query_text(raw, drop)
        scope = list(cams or self.cameras)
        method = getattr(self, "_visual_method", None) or settings["chat_method"]
        window, want = self._video_window(w, scope), int(settings["top_k"]) * 2
        where = (f" on {', '.join(self.camera_name(c) for c in cams)}" if cams and len(self.cameras) > 1 else "")
        head = f"'{text}'{where}{self._window_text(w)}{note}"
        hits, detector, how = None, None, ""
        if method.startswith("ground_"):                  # retrieve with CLIP, then detect (ovdetect.py)
            import ovdetect
            backend = method.split("_", 1)[1]
            ov = ovdetect.load_settings()
            try:
                term = ovdetect.vocab_term(text)
                vocab = ovdetect.load_vocab_index(ovdetect.vocab_index_path(self.db_path, backend)) if term else None
                if term and vocab:                        # one vocabulary word: the index built at ingest answers it
                    hits = ovdetect.vocab_search(vocab, term, cams=scope, window=window, top_k=want)
                    how = f"{DETECTOR_NAMES.get(backend, backend)} fixed-vocabulary index"
                else:
                    hits = ovdetect.ground(self.visual_index(), text, self.cameras, backend=backend, cams=scope,
                                           window=window, top_k=want, echo=lambda *a: None)
                    how = (f"{DETECTOR_NAMES.get(backend, backend)} on the {ov['candidates_k']} frames CLIP ranked "
                           f"most similar")
                detector = backend
            except Exception as exc:                      # a broken detector must not break the chat: say so
                hits, how = None, f"the detector failed ({type(exc).__name__}); CLIP frame ranking was used instead"
                method = "baseline_frame"
            if detector and not hits:
                return self._result(f"No confident match for {head}: {how} found no '{text}' box with confidence >= "
                                    f"{float(ov['conf']):.2f}. That is not proof it is absent: zero-shot detectors "
                                    f"miss small or hidden things.", "no_evidence", used=used)
        if hits is None:
            hits = openvocab.search(self.visual_index(), text, settings, method=method, cams=scope, window=window,
                                    top_k=want)
        # only a calibrated score can say "not there"; a raw similarity always ranks something first
        threshold = float(settings["min_confidence"]) if method in openvocab.CALIBRATED_METHODS else None
        if not hits or (threshold is not None and hits[0]["score"] < threshold):
            best = (f" The closest was {self._visual_where(hits[0])} with confidence {hits[0]['score']:.2f}, below "
                    f"the {threshold:.2f} needed." if hits else "")
            return self._result(f"No confident match for {head}.{best} That is not proof it is absent: the vision "
                                f"model (CLIP, zero-shot) may miss small or hidden things.", "no_evidence", used=used)
        kept, seen_gids = [], set()
        same_moment = float(settings["same_moment_s"])
        for h in hits:
            if threshold is not None and h["score"] < threshold:
                continue
            gid = self._global_id(h["camera_id"], h["track_id"]) if h["track_id"] is not None else None
            if gid and gid in seen_gids:                 # one line per swimmer: geometry-linked views are merged
                continue
            seen_gids.add(gid)
            twin = next((k for k in kept if h["track_id"] is None and k["track_id"] is None
                         and abs(k["t"] - h["t"]) < same_moment), None)
            if twin is not None:                         # the same moment seen by another (overlapping) camera
                twin.setdefault("also", []).append(h)
                continue
            kept.append(h)
        lines, cites = [], []
        out_dir = self.db_path.parent
        word = "detector confidence" if detector else "confidence" if threshold is not None else "similarity"
        for h in kept[:int(settings["top_k"])]:
            cam = self.cameras.get(h["camera_id"]) or {}
            line = f"- {self._visual_where(h)}: {word} {h['score']:.2f}"
            if h.get("also"):
                line += "; same moment also on " + ", ".join(f"{o['camera_id']} ({word} {o['score']:.2f})"
                                                           for o in h["also"])
            if h["track_id"] is not None:
                a, b = h["span"]
                line += f", matched {a:.1f}-{b:.1f} s" if b > a else ""
                partners = self._geometry_partners(h["camera_id"], h["track_id"])
                if partners:
                    line += ("; same swimmer as " + ", ".join(identity.display_track(p["track"], p["camera"])
                                                              for p in partners) + " (verified_automated: geometry)")
                alerts = self._events([h["camera_id"]], None, track=int(h["track_id"]))
                if alerts:
                    line += "; alerts on this track: " + "; ".join(
                        f"{behaviour_text(e).split(' (')[0]} at "
                        f"{_vtime(e.get('alert_s') if e.get('alert_s') is not None else e['start_s'])}" for e in alerts)
            if len(lines) < VISUAL_SHOWN:
                lines.append(line)
            cites.append({"camera_id": h["camera_id"], "camera_name": self.camera_name(h["camera_id"]),
                          "source": cam.get("video"), "track_id": h["track_id"], "event": None,
                          "event_state": ("open-vocabulary detection (box)" if detector else "open-vocabulary match"
                                          + (" (whole frame)" if h["track_id"] is None else "")),
                          "time_s": h["t"], "wall_time": openvocab.wall_text(cam, h["t"]),
                          "snapshot": openvocab.write_evidence(h, cam, out_dir, text), "confidence": h["score"],
                          "evidence": (f"{DETECTOR_NAMES.get(detector, detector)} box, {word} {h['score']:.2f}, "
                                       f"for '{text}'" if detector else
                                       f"CLIP {word} {h['score']:.2f} for '{text}' ({method})"),
                          "verification": (f"zero-shot detector box ({DETECTOR_NAMES.get(detector, detector)}): "
                                           f"not verified" if detector else
                                           "similarity only (zero-shot CLIP): not verified")})
        more = f"\n({len(cites) - len(lines)} more in Sources.)" if len(cites) > len(lines) else ""
        if detector:
            disclaimer = (f"Found by {how} (zero-shot detection, not a verified identification): the box is drawn on "
                          f"each evidence image.")
            return self._result(f"Best matches for {head}, highest detector confidence first:\n" + "\n".join(lines)
                                + more + f"\n{disclaimer}", "answered", cites, used=used)
        caveat = ("" if threshold is not None else " The ranking always puts some frame first, so it cannot prove "
                  "the thing is there: look at the images.")
        if how:                                           # the detector failed and CLIP was used instead
            caveat += f" Note: {how}."
        return self._result(f"Best matches for {head}, most similar first:\n" + "\n".join(lines) + more
                            + f"\n{openvocab.DISCLAIMER}{caveat}", "answered", cites, used=used)

    def _visual_where(self, hit: dict) -> str:
        """'15:00:17.0 (cam01 video 00:17.0) CAM01 Left wall, Track #5 (cam01)' (or 'whole frame')."""
        import openvocab
        cam = hit["camera_id"]
        who = self.who(hit["track_id"], cam) if hit["track_id"] is not None else "whole frame"
        wall = openvocab.wall_text(self.cameras.get(cam) or {}, hit["t"])
        if self.single():
            return f"{_vtime(hit['t'])}, {who}"
        return f"{_clock(wall)} ({cam} video {_vtime(hit['t'])}) {self.camera_name(cam)}, {who}"

    def _cameras_answer(self) -> dict:
        """Cameras, their clocks and their alignment status."""
        lines = []
        for cam_id, cam in self.cameras.items():
            info = camreg.sync_info(cam)
            sync = (NO_ALIGNMENT if info["sync_status"] == camreg.SYNC_UNAVAILABLE else
                    f"{info['sync_status']} (offset {info['offset_s']:+.2f} s)")
            places = [r["label"] for r in self.memory.all() if r["camera_id"] == cam_id]
            lines.append(f"- {cam.get('name')} ({cam_id}): starts {_clock(cam.get('start_time'))}, "
                         f"{float(cam.get('duration_s') or 0):.1f} s, {cam.get('n_events') or 0} event(s); time sync: "
                         f"{sync}" + (f"; your places: {', '.join(places)}" if places else ""))
        return self._result(f"{len(lines)} camera(s):\n" + "\n".join(lines), "answered" if lines else "no_evidence")

    def _events_answer(self, cams, w, behavior, note, used) -> dict:
        """What happened (filtered)."""
        events = self._events(cams, w, behavior)
        where = (f" on {', '.join(self.camera_name(c) for c in cams)}" if cams and len(self.cameras) > 1 else
                 " on any camera" if len(self.cameras) > 1 else "")
        kind = BEHAVIOUR_TEXT.get(behavior, "event").split(" (")[0] if behavior else "event"
        if not events:
            return self._result(f"No {kind} was indexed{where}{self._window_text(w)}{note}. That only means the "
                                f"pipeline raised nothing there; it is not proof that nothing happened. Ask 'which "
                                f"swimmer is in trouble?' to see the tracks that came closest.",
                                "no_evidence", used=used)
        lines = [f"- {self._event_line(e)}" for e in events]
        return self._result(f"{len(events)} {kind}(s){where}{self._window_text(w)}{note}:\n" + "\n".join(lines),
                            "answered", [self._citation(e) for e in events], used=used)

    def _risk_answer(self, cams, w, note, used) -> dict:
        """Which camera / track had the highest risk (from the risk at each indexed alert)."""
        events = [e for e in self._events(cams, w) if e.get("confidence") is not None]
        if not events:
            return self._result(f"No alert with a risk score was indexed{self._window_text(w)}{note}.", "no_evidence",
                                used=used)
        top = max(events, key=lambda e: float(e["confidence"]))
        per_cam: dict[str, float] = {}
        for e in events:
            per_cam[e["camera_id"]] = max(per_cam.get(e["camera_id"], 0.0), float(e["confidence"]))
        others = ", ".join(f"{c} {r:.0%}" for c, r in sorted(per_cam.items(), key=lambda kv: -kv[1]))
        text = (f"The highest risk at an alert{self._window_text(w)}{note} was {float(top['confidence']):.0%}: "
                f"{self._event_line(top)}." + (f"\nHighest per camera: {others}." if len(per_cam) > 1 else ""))
        return self._result(text, "answered", [self._citation(top)], used=used)

    def _dashboard(self, cam: str) -> dict | None:
        """pool_dashboard.json of a camera's run folder (risk curves, thresholds), or None."""
        out_dir = (self.cameras.get(cam) or {}).get("out_dir")
        if not out_dir:
            return None
        path = Path(out_dir) if Path(out_dir).is_absolute() else camreg.ROOT / out_dir
        return _read_json(path / "pool_dashboard.json") or None

    def _who_answer(self, cams, w, note, used) -> dict:
        """Which track is in trouble: the tracks the rules flagged, else the closest ones (never a guess)."""
        events = self._events(cams, w)
        if events:
            rank = {"submersion": 3, "aquatic_distress": 2, "aquatic_warning": 1}
            worst: dict = {}
            for e in events:
                key = (e["camera_id"], e.get("person_id"))
                if key not in worst or (rank.get(e["behavior"], 0), e.get("confidence") or 0) > (
                        rank.get(worst[key]["behavior"], 0), worst[key].get("confidence") or 0):
                    worst[key] = e
            flagged = sorted(worst.values(), key=lambda e: (-rank.get(e["behavior"], 0), -(e.get("confidence") or 0)))
            lines = [f"- {self._event_line(e)}" for e in flagged]
            return self._result(
                f"The behaviour rules flagged {len(flagged)} {'swimmer(s)' if self.single() else 'track(s)'}"
                f"{self._window_text(w)}{note}, most urgent first. "
                f"An alert means the movement matched distress signs; it does not confirm a drowning:\n" + "\n".join(lines),
                "answered", [self._citation(e) for e in flagged], used=used)
        rows = []
        scope = list(cams or self.cameras)
        vw = self._video_window(w, scope)                 # video seconds per camera, for video AND clock windows
        for cam in scope:
            if vw is not None and cam not in vw:
                continue                                  # no clock for this camera: its times cannot be placed
            start, end = vw[cam] if vw is not None else (None, None)
            dash = self._dashboard(cam) or {}
            for pid, curve in (dash.get("risk") or {}).items():
                curve = [(t, r) for t, r in curve if (start is None or t >= start) and (end is None or t <= end)]
                if curve:
                    t_peak, r_peak = max(curve, key=lambda tr: tr[1])
                    rows.append((float(r_peak), float(t_peak), cam, int(pid), dash.get("thresholds") or {}))
        if not rows:
            return self._result(f"No track was flagged and no risk data was indexed{self._window_text(w)}{note}, so "
                                f"I cannot point to anyone.", "no_evidence", used=used)
        rows.sort(key=lambda row: -row[0])
        th = rows[0][4]
        lines, cites = [], []
        for r_peak, t_peak, cam, pid, _ in rows[:CLOSEST_TRACKS]:
            lines.append(f"- {self.who(pid, cam)}: highest risk {r_peak:.0%} at "
                         f"{_vtime(t_peak) if self.single() else f'{cam} video {_vtime(t_peak)}'}")
            cites.append({"camera_id": cam, "camera_name": self.camera_name(cam), "source": None, "track_id": pid,
                          "event": None, "event_state": "no alert (peak risk only)", "time_s": t_peak,
                          "wall_time": None, "snapshot": None, "confidence": r_peak, "evidence": None,
                          "verification": "local track only: no verified cross-camera identity"})
        levels = (f" (early warning at {float(th['warning']):.0%}, distress alarm at {float(th['alert']):.0%})"
                  if th.get("warning") is not None and th.get("alert") is not None else "")
        above = th.get("warning") is not None and rows[0][0] >= float(th["warning"])
        closest = (f"Some tracks went above the warning level{levels} for a moment, but no alert was raised (the signs "
                   f"did not last long enough for the rules)" if above else
                   f"The closest tracks stayed below the warning level{levels}")
        text = (f"No track was flagged as in distress{self._window_text(w)}{note}: no alert was raised, so I cannot "
                f"say anyone was drowning. {closest}:\n"
                + "\n".join(lines) + "\nIf you see someone in trouble at those times, the detector or the rules missed "
                "it (no detection, or the movement did not match the distress signs).")
        return self._result(text, "no_evidence", cites, used=used)

    def _count_answer(self, cams, w, note, used) -> dict:
        """How many tracks were in view (per camera; merged across cameras only through geometry links)."""
        since = until = None
        if w and w["kind"] == "wall":
            since, until = (_wall_text(w["start"]) if w["start"] else None), (_wall_text(w["end"]) if w["end"] else None)
        lines, members = [], []
        for cam in cams or list(self.cameras):
            tracks = self.index.query_tracks(camera_id=cam, since_wall=since, until_wall=until)
            if w and w["kind"] == "video":
                tracks = [t for t in tracks if (w["start"] is None or float(t["last_s"]) >= w["start"])
                          and (w["end"] is None or float(t["first_s"]) <= w["end"])]
            tracks = [t for t in tracks if t.get("swimmer") in (None, 1)]
            members += [(cam, int(t["track_id"])) for t in tracks]
            lines.append(f"- {self.camera_name(cam)}: {len(tracks)} track(s)")
        if self.single():                                 # one video: a plain sentence
            ids = ", ".join(f"#{t}" for _, t in sorted(members, key=lambda m: m[1])[:12])
            span = self._window_text(w).replace(" in video time", " between").replace("-", " and ", 1) or " in the whole video"
            return self._result(f"{len(members)} swimmer(s) were tracked{span}{note}" + (f": {ids}" if ids else "")
                                + ". These are tracks, not people: one swimmer can be split into several tracks "
                                  "when they are hidden for a moment.", "answered" if members else "no_evidence",
                                used=used)
        gids = {self._global_id(c, t) for c, t in members}
        merged = len([1 for c, t in members if self._global_id(c, t) is None]) + len(gids - {None})
        text = (f"Tracks in view{self._window_text(w) or ' during the whole video'}{note}:\n" + "\n".join(lines)
                + "\nThese are tracks, not people: the tracker can split one swimmer into several tracks.")
        if len(cams or self.cameras) > 1:
            text += (f" Merging tracks that known camera geometry links across views leaves {merged} "
                     f"(of {len(members)}); unlinked tracks may still be the same swimmer.")
        return self._result(text, "answered" if members else "no_evidence", used=used)

    def _evidence_answer(self, cams, w, track, note, used) -> dict:
        """Evidence (sentence + snapshot) of the events of a track or of the filtered scope."""
        events = self._events(cams, w, track=track)
        who = self.who(track, cams[0]) if track is not None and cams else ("this video" if self.single() else "this scope")
        if not events:
            seen = self.index.query_tracks(camera_id=cams[0], track_id=track) if track is not None and cams else []
            peak = seen[0].get("peak_risk") if seen else None
            risk_txt = ("no risk score was computed for it (not judged)" if peak is None
                        else f"peak risk {float(peak):.0%}")
            extra = (f" It was tracked from {float(seen[0]['first_s']):.1f} to {float(seen[0]['last_s']):.1f} s, "
                     f"{risk_txt}, without an alert." if seen else "")
            return self._result(f"No event was indexed for {who}{self._window_text(w)}{note}.{extra}", "no_evidence",
                                used=used)
        lines = []
        for e in events:
            lines.append(f"- {self._event_line(e)}\n  Evidence: {e.get('evidence') or '(none recorded)'}"
                         + (f"\n  Snapshot: {e['snapshot']}" if e.get("snapshot") else ""))
        return self._result(f"Evidence for {who}{self._window_text(w)}{note}:\n" + "\n".join(lines), "answered",
                            [self._citation(e) for e in events], used=used)

    def _identity_answer(self, cams, track, note, used) -> dict:
        """Cross-camera continuity of a track, or the same-swimmer events found by geometry."""
        if track is None or not cams:
            groups = self.index.get_meta("geometry_same_swimmer_events") or []
            if not groups:
                return self._result(f"{identity.NO_IDENTITY_EVIDENCE} Ask about one track, e.g. 'is track 6 on cam01 "
                                    f"also seen on another camera?'", "no_evidence")
            lines = []
            for g in groups:
                parts = "; ".join(f"{e['display']} {behaviour_text(e).split(' (')[0]} "
                                  f"{float(e['start_s']):.1f}-{float(e['end_s']):.1f} s" for e in g["events"])
                lines.append(f"- {parts} (verified_automated: known camera geometry; {g['evidence'][0]})")
            return self._result(f"Alerts that are the same swimmer on several cameras{note}:\n" + "\n".join(lines),
                                "answered", used=used)
        cam = cams[0]
        who = identity.display_track(track, cam)
        if not self.index.query_tracks(camera_id=cam, track_id=track):
            return self._result(f"There is no {who} in the index.", "no_evidence")
        lines = []
        partners = self._geometry_partners(cam, track)
        for p in partners:
            ev = p["evidence"]
            lines.append(f"- {identity.display_track(p['track'], p['camera'])}: {p['status']} ({p['method']}: median "
                         f"box overlap {float(ev.get('median_iou') or 0):.2f} over {ev.get('shared_frames', '?')} "
                         f"shared frames, {float(ev.get('first_s') or 0):.1f}-{float(ev.get('last_s') or 0):.1f} s)")
        gid = self._global_id(cam, track)
        candidates = [c for c in self.index.candidates_for(cam, track)
                      if not any(c.get("camera_id") == p["camera"] and int(c.get("track_id", -1)) == int(p["track"])
                                 for p in partners)][:3]
        for c in candidates:
            lines.append(f"- {identity.candidate_text(c)}")
        if not lines:
            return self._result(f"{who}: {identity.NO_IDENTITY_EVIDENCE}", "no_evidence", used=used)
        head = f"{who}{note}" + (f" - global id {gid} (from the geometry links)" if gid else " - no global id")
        return self._result(head + ":\n" + "\n".join(lines), "answered", used=used)


FOLDER_CAMERA_ID = "cam01"
FOLDER_START = "2026-01-01 00:00:00"            # midnight: the clock of a single video then reads as its video time


def folder_conversation(folder: str | Path) -> SiteConversation:
    """Ask the Video for ONE results folder (single-camera view).

    The folder is indexed as one camera into <folder>/ask_index.sqlite3 (re-indexed when events.json changes); places
    the user defines go to <folder>/references.sqlite3. Clock times start at 00:00:00, so they equal video time.
    """
    folder = Path(folder)
    events = folder / "events.json"
    db = folder / "ask_index.sqlite3"
    index = EventIndex(db)
    stamp = round(events.stat().st_mtime, 3)
    if index.get_meta("events_mtime") != stamp:
        meta = (_read_json(events).get("meta") or {})
        video = str(meta.get("video") or "")
        camera = {"id": FOLDER_CAMERA_ID, "name": f"This video ({Path(video).name or folder.name})",
                  "video": video, "video_path": video, "start_time": FOLDER_START, "preset": None, "zones": None,
                  "offset_s": None, "sync_status": camreg.SYNC_UNAVAILABLE}
        index.index_run(camera, folder)
        index.set_meta("site", folder.name)
        index.set_meta("events_mtime", stamp)
    return SiteConversation(db, folder / "references.sqlite3")


def _read_json(path: Path) -> dict:
    """A JSON file as a dict ({} when unreadable)."""
    import json
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    """CLI: answer the questions given on the command line, in order (state is kept between them)."""
    parser = argparse.ArgumentParser(description="AquaWatch Ask the Video (grounded in the event index)")
    parser.add_argument("--db", required=True, help="event index written by ingest.py")
    parser.add_argument("--references", help="user reference store (default: references.sqlite3 next to --db)")
    parser.add_argument("questions", nargs="+")
    args = parser.parse_args(argv)
    engine = SiteConversation(args.db, args.references)
    for question in args.questions:
        result = engine.answer(question)
        text = f"Q: {question}\n[{result['status']}] {result['answer']}\n"
        print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
