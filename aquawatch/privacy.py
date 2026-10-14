"""Privacy mode: blur faces (or whole bodies) in every rendered output.

Detection, tracking and the pose check always run on the ORIGINAL frames; this module is only
used by render.py / highlights.py, on the picture that is about to be saved, BEFORE any overlay
(boxes, labels, zones) is drawn on it. The overlays are drawn on the blurred picture, so a
reviewer still sees who-did-what but never sees a face.

    privacy.is_enabled(cfg)                  -> bool
    privacy.anonymize(frame, boxes, cfg)     -> frame  (changed in place, and returned)

Config (config.yaml, section `privacy`):
    enabled        off by default (the user's choice, 2026-10-13); run.py --privacy turns it on for one run
    mode           head = blur the top `head_fraction` of every person box; body = the whole box
    head_fraction  how much of the box height counts as "head" (default 0.28)
    style          pixelate (big square blocks) | blur (strong Gaussian blur)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

# Used when the config has no `privacy` section (an old config dict, a unit test, ...).
DEFAULTS = {"enabled": False, "mode": "head", "head_fraction": 0.28, "style": "pixelate"}
DEFAULT_DOWN_ASPECT = 1.2      # width / height at or above this = person lying down (see config fall.down_min_aspect)


def _settings(cfg: dict | None) -> dict:
    """The privacy settings with defaults filled in."""
    merged = dict(DEFAULTS)
    merged.update((cfg or {}).get("privacy") or {})
    return merged


def is_enabled(cfg: dict | None) -> bool:
    """True when privacy mode is switched on (privacy.enabled in the config, or --privacy)."""
    return bool(_settings(cfg)["enabled"])


def privacy_region(box, frame_shape, cfg: dict | None) -> tuple[int, int, int, int] | None:
    """The part of one person box that gets hidden: (x0, y0, x1, y1) in whole pixels, x1 / y1 exclusive.

    The region is clamped to the frame. Returns None when nothing of the box is inside the frame
    (or the box has no usable numbers), so callers can simply skip it.
    """
    try:
        x1, y1, x2, y2 = (float(v) for v in list(box)[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    width, height = x2 - x1, y2 - y1

    st = _settings(cfg)
    whole = str(st["mode"]).lower() == "body"
    down_aspect = float((((cfg or {}).get("behaviors") or {}).get("fall") or {}).get("down_min_aspect",
                                                                                      DEFAULT_DOWN_ASPECT))
    if not whole and height > 0 and width >= down_aspect * height:
        whole = True               # a person lying down: we cannot tell where the head is, hide the whole box
    if not whole:
        fraction = min(1.0, max(0.05, float(st["head_fraction"])))
        y2 = y1 + fraction * height

    frame_h, frame_w = frame_shape[:2]
    x0, y0 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x_end, y_end = min(frame_w, int(math.ceil(x2))), min(frame_h, int(math.ceil(y2)))
    if x_end <= x0 or y_end <= y0:
        return None
    return x0, y0, x_end, y_end


def _pixelate(roi: np.ndarray) -> np.ndarray:
    """Replace a region by big flat blocks (about 4 blocks across its smaller side)."""
    h, w = roi.shape[:2]
    block = max(2, int(round(min(h, w) / 4.0)))
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST).reshape(roi.shape)


def _blur(roi: np.ndarray) -> np.ndarray:
    """Strong Gaussian blur whose strength grows with the region size (so small and big heads both vanish)."""
    h, w = roi.shape[:2]
    sigma = max(2.0, 0.3 * max(h, w))
    return cv2.GaussianBlur(roi, (0, 0), sigmaX=sigma).reshape(roi.shape)


def anonymize(frame: np.ndarray, boxes, cfg: dict | None) -> np.ndarray:
    """Hide the head (or whole body) of every person box. Works in place and returns `frame`.

    `boxes` is any iterable of (x1, y1, x2, y2) in the pixel coordinates of THIS frame (scale them
    first if the frame is bigger or smaller than the processing size). Boxes outside the frame, empty
    boxes and boxes with NaN are skipped; boxes that stick out of the frame are clamped to it.
    This function always blurs; callers decide whether to call it with `is_enabled(cfg)`.
    """
    if frame is None or getattr(frame, "size", 0) == 0 or boxes is None:
        return frame
    pixelate = str(_settings(cfg)["style"]).lower() != "blur"
    for box in boxes:
        region = privacy_region(box, frame.shape, cfg)
        if region is None:
            continue
        x0, y0, x1, y1 = region
        roi = frame[y0:y1, x0:x1]
        frame[y0:y1, x0:x1] = _pixelate(roi) if pixelate else _blur(roi)
    return frame
