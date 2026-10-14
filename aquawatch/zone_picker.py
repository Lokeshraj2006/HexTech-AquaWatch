"""Draw the restricted zone on the first frame of a video and save it as zones.json (DESIGN 3.9).

Three ways to use it:

  GUI (laptop with a screen)
      python zone_picker.py --video samples/clip.mp4 [--out zones.json] [--name restricted]
      Left-click adds a corner, right-click undoes, N starts another zone,
      Enter or S saves, Esc quits.

  Grid (no screen, e.g. Colab / SSH)
      python zone_picker.py --video samples/clip.mp4 --grid
      Writes zone_grid.jpg: the first frame with a labelled coordinate grid every 50 px.
      Open it, read off the corner coordinates of the zone ...

  Points (no screen)
      python zone_picker.py --video samples/clip.mp4 --points "x1,y1 x2,y2 x3,y3 x4,y4"
      ... and write zones.json directly. Also writes zone_preview.jpg to check the polygon.
      Several zones: separate them with ";".

All coordinates are in ORIGINAL video pixels. The pipeline scales them to the processing
frame itself (utils.load_zones). The grid / preview images are written next to --out.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

import utils
from tracker import open_video

GRID_STEP_PX = 50                          # grid line spacing, in original-frame pixels
MAX_WIN_W, MAX_WIN_H = 1280, 720           # the GUI shrinks big frames to fit a normal screen
ZONE_BGR = utils.BEHAVIOR_COLORS_BGR["zone_intrusion"]      # finished zones: red
DRAFT_BGR = (0, 215, 255)                  # the zone being drawn: amber
WINDOW_TITLE = "Draw restricted zone"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- helpers

def read_first_frame(video_path):
    """First frame of the video at ORIGINAL resolution -> (frame, (W, H)). Raises RuntimeError."""
    cap, _fps, (w, h), _n = open_video(video_path)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read the first frame of '{video_path}'.")
    return frame, (frame.shape[1], frame.shape[0])


def zone_name(base: str, index: int) -> str:
    """'restricted' for the first zone, 'restricted2', 'restricted3', ... for the others."""
    return base if index == 0 else f"{base}{index + 1}"


def polygon_area(points) -> float:
    """Area of a polygon from its corner list (shoelace formula)."""
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def parse_points(text: str, base_name: str = "restricted") -> list[dict]:
    """Parse '--points' text into zones.

    "x1,y1 x2,y2 x3,y3" is one zone; separate several zones with ';'. Any non-number
    characters (commas, spaces, brackets) are just separators. Raises ValueError.
    """
    zones = []
    for chunk in re.split(r"[;|]", text or ""):
        nums = _NUMBER.findall(chunk)
        if not nums:
            continue
        if len(nums) % 2:
            raise ValueError(f"odd number of coordinates ({len(nums)}) in '{chunk.strip()}': "
                             "each point needs an x and a y")
        pts = [(float(nums[i]), float(nums[i + 1])) for i in range(0, len(nums), 2)]
        if len(pts) < 3:
            raise ValueError(f"a zone needs at least 3 points, got {len(pts)}")
        if polygon_area(pts) < 1.0:
            raise ValueError("these points are in a straight line (the zone has no area)")
        zones.append({"name": zone_name(base_name, len(zones)), "points": pts})
    if not zones:
        raise ValueError("no coordinates found")
    return zones


def _font_scale(w: int, h: int) -> float:
    """Text size that stays readable on small and large frames."""
    return max(0.55, min(w, h) / 800.0)


def _text_size(text: str, scale: float) -> tuple[int, int]:
    """Width and height in pixels of `text` drawn with _put_text at this scale."""
    import cv2

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(round(scale))))
    return tw, th + baseline


def _put_text(img, text, org, scale, color=(255, 255, 255)):
    """Text on a darkened box so it is readable on any background (org = bottom-left of the text)."""
    import cv2

    thick = max(1, int(round(scale)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    x0, y0 = max(x - 2, 0), max(y - th - 2, 0)
    x1, y1 = min(x + tw + 2, img.shape[1]), min(y + base + 1, img.shape[0])
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = (img[y0:y1, x0:x1] * 0.35).astype(img.dtype)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _label_pos(img, text, x, y, scale):
    """Where to put a label next to point (x, y): below-right, moved inside the image if it would stick out."""
    tw, th = _text_size(text, scale)
    h, w = img.shape[:2]
    lx, ly = x + 8, y + th + 6
    if ly > h - 3:                      # no room below: put it above the point
        ly = y - 10
    if lx + tw > w - 3:                 # no room on the right: put it to the left of the point
        lx = x - tw - 8
    return int(max(lx, 2)), int(max(ly, th))


def _draw_polygon(img, pts, color, closed=True, alpha=0.30, label=None, font=0.5):
    """Draw one polygon (display pixels): translucent fill, outline, numbered corners, optional name tag."""
    import cv2

    if len(pts) == 0:
        return
    arr = np.round(np.asarray(pts, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
    if closed and len(arr) >= 3 and alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [arr], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    line_w = max(2, int(round(font * 2)))                        # thicker lines / dots on big frames
    dot_r = max(5, int(round(font * 6)))
    if len(arr) >= 2:
        cv2.polylines(img, [arr], closed and len(arr) >= 3, color, line_w, cv2.LINE_AA)
    for i, (x, y) in enumerate(arr.reshape(-1, 2)):
        cv2.circle(img, (int(x), int(y)), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), dot_r + 1, (255, 255, 255), 1, cv2.LINE_AA)
        _put_text(img, str(i + 1), _label_pos(img, str(i + 1), int(x), int(y) - 20, font * 0.9), font * 0.9)
    if label:                                                    # name tag in the middle of the zone
        cx, cy = arr.reshape(-1, 2).mean(axis=0)
        tw, th = _text_size(label, font)
        h, w = img.shape[:2]
        x0 = int(min(max(cx - tw / 2, 6), max(w - tw - 6, 6)))
        y0 = int(min(max(cy + th / 2, th + 6), h - 8))
        cv2.rectangle(img, (x0 - 4, y0 - th - 2), (x0 + tw + 4, y0 + 6), color, -1)
        cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255),
                    max(1, int(round(font))), cv2.LINE_AA)


def draw_zones(img, zones, scale: float = 1.0, coords: bool = False):
    """Draw finished zones on `img` in place. Zone points are original px; `scale` maps to display px."""
    font = _font_scale(img.shape[1], img.shape[0])
    for z in zones:
        pts = [(x * scale, y * scale) for x, y in z["points"]]
        _draw_polygon(img, pts, ZONE_BGR, closed=True, alpha=0.30, label=z["name"], font=font)
        if coords:                                               # print the corner coordinates (original px)
            for (ox, oy), (dx, dy) in zip(z["points"], pts):
                text = f"({ox:.0f},{oy:.0f})"
                _put_text(img, text, _label_pos(img, text, int(dx), int(dy), font * 0.8), font * 0.8)


def make_grid_image(frame, step: int = GRID_STEP_PX):
    """First frame with a labelled coordinate grid (lines every `step` px of the original frame).

    Lines at multiples of 2*step are yellow. Labels sit on all four edges; on very large frames
    only every few lines are labelled so the numbers never overlap.
    """
    import cv2

    h, w = frame.shape[:2]
    font = _font_scale(w, h)
    yellow, white = (0, 255, 255), (255, 255, 255)
    overlay = frame.copy()
    for x in range(0, w, step):
        cv2.line(overlay, (x, 0), (x, h - 1), yellow if x % (2 * step) == 0 else white, 1)
    for y in range(0, h, step):
        cv2.line(overlay, (0, y), (w - 1, y), yellow if y % (2 * step) == 0 else white, 1)
    img = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    tw, th = _text_size(str(max(w, h)), font)
    every = max(1, int(np.ceil((tw + 8) / step)))                # label every Nth line so labels fit
    last_end = -1                                                # right end of the previous x label
    for i, x in enumerate(range(0, w, step)):                    # x labels: top and bottom edges
        if i % every:
            continue
        lw = _text_size(str(x), font)[0]
        lx = min(x + 3, max(w - lw - 4, 0))                      # keep the last label inside the image
        if lx < last_end + 4:                                    # would overlap the previous label: skip it
            continue
        _put_text(img, str(x), (lx, th + 2), font)
        _put_text(img, str(x), (lx, h - 6), font)
        last_end = lx + lw
    for i, y in enumerate(range(0, h, step)):                    # y labels: left and right edges
        if i % every == 0 and y > 0:
            label = str(y)
            _put_text(img, label, (3, y - 3), font)
            _put_text(img, label, (max(w - _text_size(label, font)[0] - 4, 0), y - 3), font)

    tw9, _ = _text_size(f"{w},{h}", font * 0.8)                  # sparse "x,y" anchors inside the picture
    anchor = 2 * step * max(2, int(np.ceil(3 * tw9 / (2 * step))))      # round numbers, well apart
    for x in range(anchor, w - step, anchor):
        for y in range(anchor, h - step, anchor):
            cv2.circle(img, (x, y), 4, yellow, -1, cv2.LINE_AA)
            _put_text(img, f"{x},{y}", (x + 6, y - 6), font * 0.8, yellow)
    caption = f"{w}x{h} px (original frame), grid every {step} px"
    cw, _ = _text_size(caption, font * 0.8)
    _put_text(img, caption, (max((w - cw) // 2, 2), 2 * th + 10), font * 0.8, yellow)
    return img


def make_preview_image(frame, zones):
    """First frame with the zone polygons and their corner coordinates drawn on it."""
    img = frame.copy()
    draw_zones(img, zones, scale=1.0, coords=True)
    return img


def _write_image(path, img) -> bool:
    """cv2.imwrite with a readable error instead of a silent False."""
    import cv2

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = bool(cv2.imwrite(str(path), img))
    if not ok:
        print(f"WARNING: could not write image {path}", file=sys.stderr)
    return ok


# --------------------------------------------------------------------------- GUI

class ZoneEditor:
    """The click / undo / draw logic of the GUI, separate from the window so it can be tested.

    Clicks arrive in display pixels and are stored in original-frame pixels (divided by `scale`).
    """

    def __init__(self, frame, name: str = "restricted"):
        import cv2

        h, w = frame.shape[:2]
        self.size = (w, h)
        self.name = name
        self.scale = min(1.0, MAX_WIN_W / w, MAX_WIN_H / h)
        self.base = frame if self.scale == 1.0 else cv2.resize(
            frame, (max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))), interpolation=cv2.INTER_AREA)
        self.zones: list[dict] = []          # finished zones (original px)
        self.current: list[tuple[float, float]] = []   # corners of the zone being drawn (original px)
        self.message = ""

    # -- editing
    def add_point(self, dx: float, dy: float) -> None:
        """Left click at display position (dx, dy)."""
        w, h = self.size
        x = min(max(dx / self.scale, 0.0), w - 1.0)
        y = min(max(dy / self.scale, 0.0), h - 1.0)
        self.current.append((x, y))
        self.message = ""

    def undo(self) -> None:
        """Right click: remove the last corner; if the zone is empty, reopen the previous zone."""
        if self.current:
            self.current.pop()
        elif self.zones:
            self.current = list(self.zones.pop()["points"])
            self.current.pop()
        self.message = ""

    def next_zone(self) -> None:
        """Close the current zone (needs >= 3 corners) and start drawing another one."""
        if len(self.current) < 3:
            self.message = "Need at least 3 points before starting another zone."
            return
        self.zones.append({"name": zone_name(self.name, len(self.zones)), "points": list(self.current)})
        self.current = []
        self.message = f"Zone {len(self.zones)} closed. Draw the next one, or press Enter to save."

    def result(self) -> list[dict]:
        """All zones to save: finished ones plus the current one if it has >= 3 corners."""
        zones = list(self.zones)
        if len(self.current) >= 3:
            zones.append({"name": zone_name(self.name, len(zones)), "points": list(self.current)})
        return zones

    # -- drawing
    def render(self, cursor=None):
        """Image to show: first frame + finished zones + the zone being drawn + help text."""
        import cv2

        canvas = self.base.copy()
        font = _font_scale(canvas.shape[1], canvas.shape[0])
        draw_zones(canvas, self.zones, self.scale)
        pts = [(x * self.scale, y * self.scale) for x, y in self.current]
        _draw_polygon(canvas, pts, DRAFT_BGR, closed=len(pts) >= 3, alpha=0.20, font=font)
        if pts and cursor is not None:                                   # rubber-band line to the mouse
            cv2.line(canvas, (int(pts[-1][0]), int(pts[-1][1])), (int(cursor[0]), int(cursor[1])), DRAFT_BGR, 1,
                     cv2.LINE_AA)

        bar_h = int(46 * font / 0.5) + 6
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, dst=canvas)
        _put_text(canvas, "Left-click: add point | Right-click: undo | N: next zone | Enter/S: save | Esc: quit",
                  (8, int(bar_h * 0.40)), font * 0.6)
        n = len(self.current)
        status = self.message or (f"Zone '{zone_name(self.name, len(self.zones))}': {n} point(s)"
                                  + ("" if n >= 3 else " (need at least 3)"))
        if cursor is not None:
            status += f"   cursor ({cursor[0] / self.scale:.0f}, {cursor[1] / self.scale:.0f}) in original px"
        _put_text(canvas, status, (8, int(bar_h * 0.85)), font * 0.6, DRAFT_BGR)
        return canvas


def _no_display_reason() -> str | None:
    """Why a window cannot be opened here, or None if it probably can."""
    if "google.colab" in sys.modules:
        return "this is Google Colab (no screen)"
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no display found (DISPLAY is not set, typical for SSH / servers / Docker)"
    return None


def _no_gui_help(video: str, reason: str) -> str:
    """The message printed when the window cannot be used."""
    return (f"Cannot open a window: {reason}.\n"
            "Use one of the no-GUI modes instead:\n"
            f"  1) python zone_picker.py --video {video} --grid\n"
            "       writes zone_grid.jpg (the first frame with a coordinate grid);\n"
            "       open it and note the x,y of each corner of the restricted area.\n"
            f"  2) python zone_picker.py --video {video} --points \"x1,y1 x2,y2 x3,y3 x4,y4\"\n"
            "       writes zones.json and zone_preview.jpg (check the polygon in the preview).")


def run_gui(frame, video: str, name: str):
    """Open the OpenCV window. Returns the list of zones, or None if the user quit / no window."""
    reason = _no_display_reason()
    if reason:
        print(_no_gui_help(video, reason), file=sys.stderr)
        return None

    import cv2

    editor = ZoneEditor(frame, name)
    cursor = [None]

    def on_mouse(event, x, y, _flags, _param):
        cursor[0] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            editor.add_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            editor.undo()

    try:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_TITLE, on_mouse)
        cv2.imshow(WINDOW_TITLE, editor.render(cursor[0]))
        cv2.waitKey(1)
    except cv2.error as exc:             # e.g. opencv-python-headless has no window support
        print(_no_gui_help(video, f"OpenCV has no GUI support here ({str(exc).strip().splitlines()[-1]})"),
              file=sys.stderr)
        return None

    zones = None
    try:
        while True:
            cv2.imshow(WINDOW_TITLE, editor.render(cursor[0]))
            key = cv2.waitKey(30) & 0xFF
            if key == 27:                                                 # Esc
                break
            if key in (13, 10, ord("s"), ord("S")):                       # Enter / S: save
                zones = editor.result()
                if zones:
                    break
                editor.message = "Need at least 3 points before saving."
            elif key in (ord("n"), ord("N")):
                editor.next_zone()
            elif key in (8, ord("u"), ord("U")):                          # Backspace / U: undo (no right button)
                editor.undo()
            try:                                                          # window closed with the X button
                if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return zones or None


# --------------------------------------------------------------------------- CLI

def _save_and_preview(zones, size, frame, out: Path, preview: Path) -> None:
    """Write zones.json (original px) and the preview image, and tell the user."""
    utils.save_zones(out, zones, size)
    print(f"Saved {len(zones)} zone(s) to {out} (original video pixels, frame {size[0]}x{size[1]}):")
    for z in zones:
        corners = " ".join(f"({x:.0f},{y:.0f})" for x, y in z["points"])
        print(f"  {z['name']}: {corners}")
    if _write_image(preview, make_preview_image(frame, zones)):
        print(f"Preview image: {preview}")
    print(f"Use it with:  python run.py --video <video> --zones {out}")


def main(argv=None) -> int:
    """Command line entry point. Returns the process exit code."""
    ap = argparse.ArgumentParser(
        description="Mark the restricted zone on the first frame of a video and save it as zones.json.")
    ap.add_argument("--video", required=True, help="video file")
    ap.add_argument("--out", default="zones.json", help="zones file to write (default zones.json)")
    ap.add_argument("--name", default="restricted", help="zone name (default 'restricted')")
    ap.add_argument("--grid", action="store_true",
                    help="no GUI: write zone_grid.jpg, the first frame with a coordinate grid every 50 px")
    ap.add_argument("--points", default=None, metavar='"x1,y1 x2,y2 x3,y3 ..."',
                    help="no GUI: write the zone from these corner points (original video pixels); "
                         "separate several zones with ';'")
    ap.add_argument("--preview", default=None, metavar="IMAGE",
                    help="where to write the zone preview image (default zone_preview.jpg next to --out)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    preview = Path(args.preview) if args.preview else out.parent / "zone_preview.jpg"
    try:
        frame, size = read_first_frame(args.video)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.grid:
        grid_path = out.parent / "zone_grid.jpg"
        if not _write_image(grid_path, make_grid_image(frame)):
            return 1
        print(f"Wrote {grid_path} ({size[0]}x{size[1]} px, grid every {GRID_STEP_PX} px).\n"
              "Open it, read the x,y of each corner of the restricted area, then run:\n"
              f"  python zone_picker.py --video {args.video} --points \"x1,y1 x2,y2 x3,y3 x4,y4\"")

    if args.points is not None:
        try:
            zones = parse_points(args.points, args.name)
        except ValueError as exc:
            print(f"ERROR in --points: {exc}.\nExpected e.g. --points \"100,200 400,200 400,350 100,350\"",
                  file=sys.stderr)
            return 1
        for z in zones:                                                  # warn about likely typos, but keep them
            if any(not (0 <= x <= size[0] and 0 <= y <= size[1]) for x, y in z["points"]):
                print(f"WARNING: zone '{z['name']}' has points outside the {size[0]}x{size[1]} frame.",
                      file=sys.stderr)
        _save_and_preview(zones, size, frame, out, preview)
        return 0

    if args.grid:
        return 0

    zones = run_gui(frame, args.video, args.name)
    if not zones:
        print("No zones saved.", file=sys.stderr)
        return 1
    _save_and_preview(zones, size, frame, out, preview)
    return 0


if __name__ == "__main__":
    sys.exit(main())
