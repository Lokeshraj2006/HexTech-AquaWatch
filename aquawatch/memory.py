"""Persistent, user-defined place references for the AquaWatch conversation ("CAM03 is the deep end").

A reference maps a place name the USER gave ("deep end") to a camera (and optionally a region of it).
It is never derived from computer vision and is never presented as CV truth: every row has
source = "user_defined" and verification_status = "user_defined", plus the chat text it came from.

SQLite tables (parameterised SQL only):
  user_references        : the current value of each reference (one row per normalised key)
  user_reference_history : every value a reference ever had (created / updated / forgotten), oldest first

    memory = ReferenceMemory("outputs/virtual_cctv/references.sqlite3")
    memory.remember("deep end", "cam03", provenance="chat: 'CAM03 is the deep end'")
    memory.lookup("deep end")      -> {"reference": "deep_end", "label": "deep end", "camera_id": "cam03", ...}
    memory.forget("deep end")      -> True (soft delete: the row stays in the history)
    parse_statement("the deep end is cam03") -> {"reference": "deep_end", "label": "deep end", "camera": "cam03", ...}

Pure Python (sqlite3 + re); no cv2 / numpy.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

SOURCE_USER = "user_defined"
STATUS_USER = "user_defined"
USER_NOTE = "User-defined reference given in chat; not derived from computer vision and not verified by it."

# words that make a phrase a question / a sentence about tracks, never a place name
_NOT_A_PLACE = re.compile(r"\b(track|tracks|swimmer|swimmers|person|people|event|events|risk|camera|cameras|cam|"
                          r"what|which|when|where|who|why|how|happened|seen|saw|show|evidence|identity|same)\b")
_QUESTION_START = re.compile(r"^(is|are|was|were|what|which|when|where|who|why|how|does|did|do|can|could|show|"
                             r"list|tell|give|find)\b")
# a camera token: "cam03", "CAM 3", "camera 3", "camera #3", "cam_03", or a registry id like "cam_deep"
CAMERA_TOKEN = r"(?:cam(?:era)?\s*[_#-]?\s*\d+|cam_[a-z0-9_]+)"
_REGION = (r"(?:the\s+)?(?P<region>(?:left|right|top|bottom|upper|lower|near|far|middle|centre|center)"
           r"(?:\s+(?:half|side|part|third|corner|edge))?)\s+of\s+")
# corrections too ("actually the deep end is cam04", "no, ...", "sorry, I meant ..."): the place itself is updated
# (audit 2026-10-13 F02: "actually" used to become part of the place name)
_LEAD_IN = re.compile(r"^(?:ok(?:ay)?[,.]?\s+|so[,.]?\s+|fyi[,:]?\s+|note(?:\s+that)?[,:]?\s+|"
                      r"remember(?:\s+that)?[,:]?\s+|please\s+remember(?:\s+that)?\s+|for\s+the\s+record[,:]?\s+|"
                      r"actually[,]?\s+|no[,.]?\s+|nope[,.]?\s+|sorry[,.]?\s+|oops[,.]?\s+|wait[,.]?\s+|"
                      r"correction[,:]?\s+|update[,:]?\s+|i\s+meant(?:\s+that)?[,:]?\s+|i\s+mean[,:]?\s+)+")
_VERB = r"(?:is|=|==|->|covers|shows|is\s+covered\s+by|is\s+seen\s+(?:by|on)|is\s+on|is\s+in|means|equals)"

# the last word of a place name ("deep END", "diving BOARD"); a phrase ending otherwise is not taken as a place
PLACE_NOUNS = frozenset((
    "end ends side sides area areas zone zones corner corners lane lanes steps stairs ladder ladders slide slides "
    "entrance entry exit edge wall walls deck gate board pool pools shallows spa jacuzzi tub fountain bay platform "
    "stand stands chair station tower region section half door ramp rail rope gutter bench bleachers block blocks "
    "island basin beach spot part parts water waters").split())
# place nouns that name a place on their own ("the slide"); the others need a modifier ("the deep end")
STANDALONE_PLACES = frozenset("slide ladder steps stairs entrance entry exit deck gate jacuzzi spa fountain shallows "
                              "bleachers tower".split())
_NOT_MODIFIERS = frozenset("same other whole entire first last video clip recording camera footage frame this that "
                           "each every any".split())
_PLACE_IN_QUESTION = re.compile(r"\b(?:at|in|near|around|by|on|from|to|of|covering|over)\s+the\s+"
                                r"((?:[a-z'-]+\s+){0,3}?[a-z'-]+)(?=\s*(?:$|[?.!,;]|\s+(?:and|or|between|at|on|in|"
                                r"during|from|before|after|since|until|camera|cam)\b|"
                                # "the deep end of the (swimming) pool": the place is "deep end" (audit F03)
                                r"\s+of\s+(?:the\s+)?(?:swimming\s+)?pool\b))")


def now_text() -> str:
    """Local wall clock now as 'YYYY-MM-DDTHH:MM:SS'."""
    return datetime.now().isoformat(timespec="seconds")


def normalize_reference(text: str) -> str:
    """'The Deep End' -> 'deep_end' (lower case, no leading article, words joined by '_')."""
    words = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    while words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return "_".join(words)


def reference_label(text: str) -> str:
    """Human label of a reference: 'deep_end' / 'The Deep End' -> 'deep end'."""
    return normalize_reference(text).replace("_", " ")


def resolve_camera_token(token: str, known_ids: Iterable[str] | None = None) -> str | None:
    """'CAM03' / 'camera 3' / 'cam 3' -> a camera id.

    With `known_ids` (the registry ids) the token must name one of them: an exact (case-insensitive) id, or the
    only id whose trailing number equals the token's number. Returns None when it names none (or several).
    Without `known_ids`: 'cam' + two-digit number ('camera 3' -> 'cam03'), else the lower-case token.
    """
    text = re.sub(r"\s+", " ", str(token).strip().lower())
    if not text:
        return None
    number_match = re.fullmatch(r"cam(?:era)?\s*[_#-]?\s*0*(\d+)", text)
    number = int(number_match.group(1)) if number_match else None
    if known_ids is None:
        return f"cam{number:02d}" if number is not None else text.replace(" ", "_")
    ids = [str(i) for i in known_ids]
    for camera_id in ids:
        if camera_id.lower() == text.replace(" ", "_"):
            return camera_id
    if number is None:
        return None
    hits = []
    for camera_id in ids:
        tail = re.search(r"(\d+)$", camera_id)
        if tail and int(tail.group(1)) == number:
            hits.append(camera_id)
    return hits[0] if len(hits) == 1 else None


# Place wording ('clarify once'): "deep part", "deep side", "the deep end of the pool" all mean the place the user
# saved as "deep end". A place is reduced to its CORE: articles, "of the pool", and generic head words dropped, a few
# plain synonyms unified. Only an exact core match counts; two saved places with one core are asked about, not guessed.
GENERIC_PLACE_WORDS = frozenset("end ends side sides part parts area areas section sections half zone zones region "
                                "regions water waters bit".split())
_CORE_SYNONYMS = {"deeper": "deep", "deepest": "deep", "shallows": "shallow", "shallower": "shallow",
                  "kid": "kids", "kiddie": "kids", "kiddies": "kids", "child": "kids", "children": "kids",
                  "childrens": "kids", "toddler": "kids", "toddlers": "kids", "centre": "center", "middle": "center"}
_POOL_TAIL = re.compile(r"\s+(?:of|in|at)\s+(?:the\s+)?(?:swimming\s+)?pool(?:\s+area)?$")


def place_core(text: str) -> str:
    """'the deep end of the pool' / 'deep part' / 'deep side' -> 'deep'; 'shallows' -> 'shallow'; '' when nothing
    specific is left ('the area', 'the water')."""
    phrase = re.sub(r"[^a-z0-9' ]+", " ", str(text).lower()).strip()
    phrase = _POOL_TAIL.sub("", re.sub(r"\s+", " ", phrase))
    words = [_CORE_SYNONYMS.get(w, w) for w in phrase.split()]
    while words and words[0] in ("the", "a", "an"):
        words = words[1:]
    return " ".join(w for w in words if w not in GENERIC_PLACE_WORDS and w != "pool")


def is_place_name(text: str) -> bool:
    """True for a short place name: 1-5 words, ends in a place noun, a lone noun only when it names a place itself."""
    words = str(text).split()
    if not words or len(words) > 5 or _NOT_A_PLACE.search(text):
        return False
    if words[-1] not in PLACE_NOUNS:
        return False
    if len(words) == 1:
        return words[0] in STANDALONE_PLACES
    return not any(word in _NOT_MODIFIERS for word in words[:-1])


def _clean_place(text: str) -> str | None:
    """A place phrase from a statement, or None when it does not look like a short place name."""
    text = re.sub(r"\s+", " ", text.strip().strip(".,;:!\"'").strip())
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    if not text or not re.fullmatch(r"[a-z][a-z' -]*", text):
        return None
    return text if is_place_name(text) else None


# a pool place typed without "the" ("any alerts in deep end?"): only these pool words, so "kids in the pool" or
# "at end of video" are never read as a place (audit 2026-10-13 F03: such a question used to cover all cameras)
_BARE_POOL_PLACE = re.compile(r"\b(?:at|in|near|around|by|on|from|covering|over)\s+((?:deep(?:er|est)?|shallow(?:er)?|"
                              r"kids?|kiddie|children'?s?|toddlers?|diving|wading)\s+(?:end|side|part|area|section|"
                              r"half|zone|water|pool|corner))\b")


def find_place_phrase(question: str) -> str | None:
    """The first place name in a question ('what happened at the deep end?' -> 'deep end'), else None."""
    lowered = re.sub(r"\s+", " ", str(question or "").lower())
    for match in _PLACE_IN_QUESTION.finditer(lowered):
        words = match.group(1).split()
        # try the longest tail first: "the busy deep end" -> "busy deep end", "deep end", ...
        for start in range(len(words)):
            phrase = " ".join(words[start:])
            if is_place_name(phrase):
                return phrase
    bare = _BARE_POOL_PLACE.search(lowered)
    return bare.group(1) if bare and is_place_name(bare.group(1)) else None


_CAMERA_REPLY = re.compile(rf"^(?:(?:it'?s|it\s+is|that'?s|that\s+is|on|use|in|try|answer|the\s+answer\s+is)\s+)?"
                           rf"(?:the\s+)?(?P<cam>{CAMERA_TOKEN})(?:\s+(?:please|thanks|thank\s+you))?$")


def camera_reply(text: str, known_ids: Iterable[str] | None = None) -> tuple[str, str | None] | None:
    """A reply that only names one camera ("cam03", "it's camera 3"): (token as typed, resolved id or None), else None."""
    lowered = _LEAD_IN.sub("", re.sub(r"\s+", " ", str(text or "").strip().lower()).rstrip(".!"))
    match = _CAMERA_REPLY.match(lowered)
    if not match:
        return None
    token = match.group("cam")
    return token.strip(), resolve_camera_token(token, known_ids)


def parse_statement(text: str, known_ids: Iterable[str] | None = None) -> dict | None:
    """Read a user statement that defines a place. Returns None for anything else (questions included).

    Understood forms (case-insensitive, optional lead-in such as 'remember that'):
      'CAM03 is the deep end' / 'camera 3 covers the deep end'
      'the deep end is cam03' / 'deep end = camera 3' / 'the deep end is the left half of cam03'
    Returns {"reference", "label", "camera" (resolved id, or None when `known_ids` has no such camera),
    "camera_token" (as typed), "region" (text or None)}.
    """
    raw = re.sub(r"\s+", " ", str(text or "").strip())
    if not raw or raw.endswith("?"):
        return None
    lowered = _LEAD_IN.sub("", raw.lower().rstrip(".!"))
    if _QUESTION_START.match(lowered):
        return None
    patterns = (
        # camera first: "cam03 is the deep end"
        rf"^(?P<cam>{CAMERA_TOKEN})\s+{_VERB}\s+(?P<place>.+)$",
        # place first, region of a camera: "the deep end is the left half of cam03"
        rf"^(?P<place>.+?)\s+{_VERB}\s+{_REGION}(?P<cam>{CAMERA_TOKEN})$",
        # place first: "the deep end is cam03" / "deep end = camera 3"
        rf"^(?P<place>.+?)\s+{_VERB}\s+(?P<cam>{CAMERA_TOKEN})$",
    )
    for pattern in patterns:
        match = re.match(pattern, lowered)
        if not match:
            continue
        place = _clean_place(match.group("place"))
        if place is None:
            continue
        token = match.group("cam")
        region = match.groupdict().get("region")
        return {"reference": normalize_reference(place), "label": reference_label(place),
                "camera": resolve_camera_token(token, known_ids), "camera_token": token.strip(),
                "region": region.strip() if region else None}
    return None


class ReferenceMemory:
    """SQLite store of user-defined place references (survives restarts; soft delete keeps the history)."""

    def __init__(self, database_path: str | Path, clock: Callable[[], str] | None = None):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or now_text
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_references (
                    reference TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    region TEXT,
                    source TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_reference_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reference TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    label TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    region TEXT,
                    provenance TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _region_text(region: Any) -> str | None:
        """Store a region as text: plain strings as they are, anything else as JSON."""
        if region in (None, ""):
            return None
        return region if isinstance(region, str) else json.dumps(region, sort_keys=True)

    @staticmethod
    def _region_value(text: str | None) -> Any:
        if text is None:
            return None
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        return value if isinstance(value, (list, dict)) else text

    def _row(self, row: sqlite3.Row) -> dict:
        record = dict(row)
        record["region"] = self._region_value(record.get("region"))
        record["active"] = bool(record.get("active"))
        record["note"] = USER_NOTE
        return record

    def remember(self, reference: str, camera_id: str, region: Any = None, provenance: str = "") -> dict:
        """Create or update a reference; returns the stored record. Updating increments `version`."""
        key = normalize_reference(reference)
        camera_id = str(camera_id or "").strip()
        if not key or not camera_id:
            raise ValueError("Both a reference name and a camera ID are required")
        label = reference_label(reference)
        provenance = str(provenance or "user_defined (no chat text recorded)")
        region_text = self._region_text(region)
        stamp = self._clock()
        connection = self._connect()
        try:
            old = connection.execute("SELECT * FROM user_references WHERE reference = ?", (key,)).fetchone()
            if old is None:
                version, created, action = 1, stamp, "created"
                connection.execute(
                    "INSERT INTO user_references (reference, label, camera_id, region, source, verification_status, "
                    "provenance, created_at, updated_at, version, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (key, label, camera_id, region_text, SOURCE_USER, STATUS_USER, provenance, created, stamp, version),
                )
            else:
                version, created = int(old["version"]) + 1, old["created_at"]
                action = "updated" if old["active"] else "restored"
                connection.execute(
                    "UPDATE user_references SET label = ?, camera_id = ?, region = ?, provenance = ?, updated_at = ?, "
                    "version = ?, active = 1 WHERE reference = ?",
                    (label, camera_id, region_text, provenance, stamp, version, key),
                )
            connection.execute(
                "INSERT INTO user_reference_history (reference, version, action, label, camera_id, region, provenance, "
                "recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, version, action, label, camera_id, region_text, provenance, stamp),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM user_references WHERE reference = ?", (key,)).fetchone()
        finally:
            connection.close()
        return self._row(row)

    def lookup(self, reference: str) -> dict | None:
        """The active record of a reference, or None (unknown or forgotten)."""
        key = normalize_reference(reference)
        if not key:
            return None
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM user_references WHERE reference = ? AND active = 1",
                                     (key,)).fetchone()
        finally:
            connection.close()
        return self._row(row) if row is not None else None

    def all(self, include_forgotten: bool = False) -> list[dict]:
        """Every active reference (or every reference), ordered by key."""
        sql = "SELECT * FROM user_references" + ("" if include_forgotten else " WHERE active = 1")
        connection = self._connect()
        try:
            rows = connection.execute(sql + " ORDER BY reference").fetchall()
        finally:
            connection.close()
        return [self._row(row) for row in rows]

    def forget(self, reference: str, provenance: str = "") -> bool:
        """Soft-delete a reference (lookup() no longer finds it; the history keeps it). False when not active."""
        key = normalize_reference(reference)
        stamp = self._clock()
        connection = self._connect()
        try:
            old = connection.execute("SELECT * FROM user_references WHERE reference = ? AND active = 1",
                                     (key,)).fetchone()
            if old is None:
                return False
            version = int(old["version"]) + 1
            connection.execute("UPDATE user_references SET active = 0, version = ?, updated_at = ? WHERE reference = ?",
                               (version, stamp, key))
            connection.execute(
                "INSERT INTO user_reference_history (reference, version, action, label, camera_id, region, provenance, "
                "recorded_at) VALUES (?, ?, 'forgotten', ?, ?, ?, ?, ?)",
                (key, version, old["label"], old["camera_id"], old["region"],
                 str(provenance or "forget()"), stamp),
            )
            connection.commit()
            return True
        finally:
            connection.close()

    def history(self, reference: str) -> list[dict]:
        """Every recorded value of a reference, oldest first (actions: created / updated / restored / forgotten)."""
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM user_reference_history WHERE reference = ? ORDER BY id",
                                      (normalize_reference(reference),)).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            record = dict(row)
            record["region"] = self._region_value(record.get("region"))
            result.append(record)
        return result

    def find_by_core(self, place: str) -> list[dict]:
        """Active references whose place CORE equals the core of `place` ('deep part' finds 'deep end')."""
        core = place_core(place)
        return [record for record in self.all() if core and place_core(record["label"]) == core]

    def find_in_text(self, text: str) -> list[dict]:
        """Active references whose label occurs (as whole words) in `text`, longest label first."""
        lowered = " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()) + " "
        found = [record for record in self.all() if f" {record['label']} " in lowered]
        return sorted(found, key=lambda record: -len(record["label"]))


def describe(record: dict) -> str:
    """One plain sentence about a stored reference, always marked as user-defined."""
    region = f" ({record['region']})" if record.get("region") else ""
    day = str(record.get("updated_at") or "")[:10]
    return (f"{record['label']} = {str(record['camera_id']).upper()}{region}, set by you on {day} "
            f"(user-defined, not detected by the cameras)")
