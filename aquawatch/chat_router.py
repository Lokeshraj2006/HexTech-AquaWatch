"""Meaning-based question routing for Ask AquaWatch: which kind of answer a question wants.

The word rules in ps_conversation.py are exact but brittle ("Which swimmer needs help?" hit the help text). This module
compares a question with example questions of every answer type, by meaning: each text is embedded with the CLIP text
encoder the chat already uses for visual search (OpenAI ViT-B/32, local file, nothing downloaded), and the type whose
two closest examples are most similar wins. It only decides when it is clearly ahead (MARGIN); otherwise it returns
None and the word rules decide. The examples below were written without looking at the test questions
(data/ground_truth/chat_questions.yaml); tools/eval_chat.py measures the result.

torch / open_clip are imported inside functions only (through openvocab).
"""
from __future__ import annotations

import threading

import numpy as np

MARGIN = 0.03          # the best type must beat the runner-up by this much cosine similarity (fixed before measuring)
TOP = 2                # a type's score = mean of its TOP closest examples
EXAMPLES: dict[str, list[str]] = {
    "events": ["what went on in this video", "tell me what happened", "list every alert", "were any warnings raised",
               "give me an overview of the incidents", "anything unusual in the footage", "show all events",
               "did the system flag anything", "recap the clip", "what occurred near the end"],
    "who": ["which person is in trouble", "is anybody in distress", "who might be drowning right now",
            "which swimmer looks like they need assistance", "is a child struggling", "who needs to be rescued",
            "which swimmer should be checked", "is someone going under", "who is at risk of drowning",
            "point me to the swimmer in difficulty",
            # drowning signs (added 2026-10-11 after a live check; not taken from the test questions)
            "is anyone floating face down", "is a swimmer lying motionless in the water"],
    "risk": ["what was the highest risk score", "which moment was the riskiest", "which camera recorded the worst risk",
             "when did the danger peak", "show the maximum risk", "what is the top risk level reached",
             "which swimmer scored the highest danger", "where was the risk greatest"],
    "count": ["how many people did you see", "number of swimmers in the water", "how many tracks were detected",
              "how busy was the pool", "total count of persons", "how full was the pool at that time",
              "tell me the number of people", "how many were swimming"],
    "evidence": ["why did you raise that alert", "show me the proof", "what evidence is there for this warning",
                 "explain why the swimmer was flagged", "show the picture of the alert", "give me the snapshot",
                 "what signs did you see", "justify the alarm"],
    "identity": ["is this the same person on another camera", "does the swimmer appear in other views",
                 "match this track across cameras", "which detections are the same swimmer",
                 "follow this person between cameras", "was this swimmer seen elsewhere"],
    "cameras": ["list the cameras", "what cameras do you have", "are the clocks of the cameras aligned",
                "how many cameras are installed", "which views are available", "camera synchronisation status"],
    "visual": ["find a man in a red cap", "search for someone holding a float", "is there a person with a black swimsuit",
               "look for a kid on an inflatable", "show frames with a blue towel", "spot a person sitting on the edge",
               "do you see anyone with goggles", "find the lifeguard chair"],
    "help": ["what can you do", "how do I use this chat", "what questions can I ask", "what do you know about",
             "explain your features", "what kinds of questions work"],
}

_CACHE: dict = {}
_LOCK = threading.Lock()   # the dashboard warms the model up in a background thread: never load it twice at once


def _encode(texts: list[str]) -> np.ndarray:
    """Unit-length CLIP text embeddings (the local OpenAI ViT-B/32, the same model as the chat's visual search)."""
    import openvocab
    settings = openvocab.load_settings({"pretrained": "openai"})
    return openvocab.embed_texts(list(texts), settings)


def _example_matrix() -> tuple[list[str], np.ndarray]:
    """(type of each example, their embeddings), encoded once per process."""
    with _LOCK:
        if "matrix" not in _CACHE:
            labels = [k for k, items in EXAMPLES.items() for _ in items]
            texts = [t for items in EXAMPLES.values() for t in items]
            _CACHE["matrix"] = (labels, _encode(texts))
    return _CACHE["matrix"]


def warm_up() -> bool:
    """Load the text model and encode the examples now (about 30 s on this laptop's CPU), so the first chat question
    is answered at once. The dashboard calls this in a background thread at start-up. False when it cannot run."""
    try:
        _example_matrix()
        return True
    except Exception:                                   # no CLIP weights here: the word rules answer instead
        return False


def scores(question: str, multi_camera: bool = True) -> dict[str, float]:
    """Score of every answer type for a question (mean cosine similarity of its TOP closest examples)."""
    labels, matrix = _example_matrix()
    sims = matrix @ _encode([question])[0]
    out: dict[str, float] = {}
    for kind in EXAMPLES:
        if not multi_camera and kind in ("identity", "cameras"):
            continue                                  # one video: no other cameras to talk about
        s = np.sort(sims[[i for i, label in enumerate(labels) if label == kind]])[::-1][:TOP]
        out[kind] = float(s.mean())
    return out


def intent(question: str, multi_camera: bool = True) -> str | None:
    """The answer type a question wants, or None when the router is not clearly sure (then the word rules decide)."""
    ranked = sorted(scores(question, multi_camera).items(), key=lambda kv: -kv[1])
    if len(ranked) < 2 or ranked[0][1] - ranked[1][1] < MARGIN:
        return None
    return ranked[0][0]
