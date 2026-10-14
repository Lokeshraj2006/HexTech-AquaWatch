"""Merge several downloaded YOLO datasets into ONE training set (our 5 body-state classes, or the recipe's `classes`).

    python training/build_dataset.py                      # uses training/pool_datasets.yaml
    python training/build_dataset.py --recipe my.yaml --limit 300   # small copy for a quick test

What it does, and why:
1. Remaps every source's class names to ours (the `map` in the recipe; "drop" removes a box).
2. Removes DUPLICATE frames: community datasets copy each other (here poolsafety is the same frames as
   university). A frame is identified by its name before Roboflow's ".rf.<hash>" suffix ("drowning1_mp4-123").
   When a name was already taken from an earlier source, the PIXELS are compared (a small image hash): only a
   real copy is skipped, because generic names like "youtube-45" can be different videos in different
   datasets. Roboflow's own augmented copies of a frame inside ONE source are kept (extra variety).
3. Splits BY SOURCE VIDEO ("drowning1_mp4", "youtube", ...), never by frame: frames of one video are near
   identical, so mixing them across train and test would make the test score look far better than it is.
   The video name is shared across sources, so a video re-uploaded in two datasets still lands in ONE split
   (even a duplicate the pixel check missed cannot leak from train into test).
4. Prefixes file names with the source, writes data.yaml, manifest.csv (every image: source, video, split)
   and CREDITS.md (the licences to declare in the README).
Files are hard-linked when possible (no extra disk space), else copied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUR_CLASSES = ["swimmer_horizontal", "swimmer_vertical", "head_only", "underwater", "out_of_water"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RF_SUFFIX = re.compile(r"(_(jpg|jpeg|png))?\.rf\.[0-9a-f]+$", re.I)


def frame_key(stem: str) -> str:
    """'drowning1_mp4-123_jpg.rf.abc' -> 'drowning1_mp4-123' (the original frame, before Roboflow's suffix)."""
    return RF_SUFFIX.sub("", stem).lower()


EXT_TAIL = re.compile(r"_(jpe?g|png|bmp|webp)$", re.I)
NUMERIC_BUCKET = 100       # bare-number frame names: 100 consecutive numbers count as one video


def video_key(stem: str) -> str:
    """'drowning1_mp4-123_jpg.rf.abc' -> 'drowning1_mp4' (the source video; frame number removed).

    Some sources name frames only by a number ('127', '128', ...) or with the extension left in ('d22_jpeg'). Those
    neighbours are frames of one video, so they must share a split: the extension tail is dropped first, and bare
    numbers are grouped in buckets of NUMERIC_BUCKET ('127' -> 'num_1')."""
    frame = EXT_TAIL.sub("", frame_key(stem))
    if re.fullmatch(r"\d+", frame):
        return f"num_{int(frame) // NUMERIC_BUCKET}"
    return re.sub(r"[-_ ]?\d+$", "", frame) or frame


def load_source(name: str, spec: dict, classes: list[str] = OUR_CLASSES) -> list[dict]:
    """Every (image, label) pair of one source with its labels remapped to `classes` (default: our 5 states)."""
    path = (ROOT / spec["path"]) if not Path(spec["path"]).is_absolute() else Path(spec["path"])
    names = yaml.safe_load(open(path / "data.yaml", encoding="utf-8"))["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    mapping = {str(k).strip().lower(): v for k, v in (spec.get("map") or {}).items()}
    lookup = {}
    for i, n in enumerate(names):
        target = mapping.get(n.strip().lower(), "drop")
        if target not in classes + ["drop"]:
            raise SystemExit(f"{name}: '{n}' -> '{target}' is not one of {classes} or drop")
        lookup[i] = None if target == "drop" else classes.index(target)
    items = []
    for split in ("train", "valid", "val", "test"):
        img_dir = path / split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            lab = path / split / "labels" / (img.stem + ".txt")
            lines = []
            if lab.exists():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    p = line.split()
                    if len(p) >= 5 and lookup.get(int(float(p[0]))) is not None:
                        lines.append(" ".join([str(lookup[int(float(p[0]))])] + p[1:5]))
            items.append({"source": name, "img": img, "lines": lines, "frame": frame_key(img.stem),
                          "video": video_key(img.stem)})   # NOT per source: one video = one split everywhere
    return items


def assign_splits(videos: dict[str, int], fractions, seed: int) -> dict[str, str]:
    """Put whole videos into train / val / test so each split gets about its share of images.
    Order is a stable hash of the video name (+ seed), so the split never changes between runs."""
    order = sorted(videos, key=lambda v: hashlib.md5(f"{seed}:{v}".encode()).hexdigest())
    total = sum(videos.values())
    want = {"test": fractions[2] * total, "val": fractions[1] * total}
    have = Counter()
    out = {}
    for v in order:
        n = videos[v]
        if have["test"] + n <= want["test"] * 1.15 and have["test"] < want["test"]:
            out[v] = "test"
        elif have["val"] + n <= want["val"] * 1.15 and have["val"] < want["val"]:
            out[v] = "val"
        else:
            out[v] = "train"
        have[out[v]] += n
    return out


_HASHES: dict[str, object] = {}


def image_hash(path: Path):
    """16x16 average hash (256 bits) of an image, cached: near-identical images have a tiny Hamming distance."""
    key = str(path)
    if key not in _HASHES:
        import cv2
        import numpy as np
        img = cv2.imread(key, cv2.IMREAD_GRAYSCALE)
        if img is None:
            _HASHES[key] = None
        else:
            g = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA).astype(float)
            _HASHES[key] = (g > g.mean()).flatten()
    return _HASHES[key]


def is_same_image(a: Path, others: list[Path], max_dist: int = 40) -> bool:
    """True if image `a` is pixel-wise the same picture as any of `others` (Hamming distance <= max_dist of 256)."""
    ha = image_hash(a)
    if ha is None:
        return False
    for o in others[:4]:
        ho = image_hash(o)
        if ho is not None and int((ha != ho).sum()) <= max_dist:
            return True
    return False


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recipe", default=str(ROOT / "training" / "pool_datasets.yaml"))
    ap.add_argument("--out", default=None, help="override the recipe's output folder")
    ap.add_argument("--limit", type=int, default=0, help="keep only N images per source (quick tests)")
    args = ap.parse_args(argv)
    recipe = yaml.safe_load(open(args.recipe, encoding="utf-8"))
    classes = list(recipe.get("classes") or OUR_CLASSES)        # e.g. [person] for a single-class detector
    out = Path(args.out or recipe["output"])
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        shutil.rmtree(out)

    kept, seen_frames, dups, name_clash = [], defaultdict(list), Counter(), Counter()
    for name, spec in recipe["sources"].items():
        if spec.get("enabled", True) is False:
            print(f"  skipping source '{name}' (enabled: false)")
            continue
        items = load_source(name, spec, classes)
        if args.limit:
            items = items[: args.limit]
        own_frames = defaultdict(list)
        for it in items:
            earlier = seen_frames.get(it["frame"])
            if earlier and it["frame"] not in own_frames:
                if is_same_image(it["img"], earlier):
                    dups[name] += 1                                # a real copy of a frame from an earlier source
                    continue
                name_clash[name] += 1                              # same name, different picture: keep it
                it["frame"] = f"{name}:{it['frame']}"
            own_frames[it["frame"]].append(it["img"])
            kept.append(it)
        for k, v in own_frames.items():
            seen_frames[k] += v

    videos = Counter(it["video"] for it in kept)
    split_of = assign_splits(videos, recipe.get("split", [0.8, 0.1, 0.1]), int(recipe.get("seed", 7)))
    boxes = defaultdict(Counter)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "video", "split", "original", "boxes"])
        for i, it in enumerate(kept):
            split = split_of[it["video"]]
            new = f"{it['source']}_{i:06d}{it['img'].suffix.lower()}"
            _place(it["img"], out / split / "images" / new)
            (out / split / "labels" / (Path(new).stem + ".txt")).write_text("\n".join(it["lines"]), encoding="utf-8")
            for line in it["lines"]:
                boxes[split][classes[int(line.split()[0])]] += 1
            w.writerow([new, it["source"], it["video"], split, it["img"].name, len(it["lines"])])
    yaml.safe_dump({"path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
                    "names": {i: n for i, n in enumerate(classes)}}, open(out / "data.yaml", "w"), sort_keys=False)
    used = {n: s for n, s in recipe["sources"].items() if s.get("enabled", True) is not False}
    credits = "\n".join(f"- {spec['credit']}" for spec in used.values() if spec.get("credit"))
    (out / "CREDITS.md").write_text("# Datasets used (declare these in the README)\n\n" + credits + "\n", encoding="utf-8")

    imgs = Counter(split_of[it["video"]] for it in kept)
    vids = Counter(split_of[v] for v in videos)
    print(f"Kept {len(kept)} images from {len(used)} sources; skipped real duplicates: {dict(dups)}; "
          f"kept same-name-but-different images: {dict(name_clash)}")
    for split in ("train", "val", "test"):
        print(f"  {split:<5} {imgs[split]:>6} images from {vids[split]:>3} videos  boxes: {dict(boxes[split])}")
    empty = [c for c in classes if sum(boxes[s][c] for s in boxes) == 0]
    if empty:
        print("  No boxes yet for:", ", ".join(empty), "(label frames from your staged clips to add them)")
    print(f"Wrote {out / 'data.yaml'}, manifest.csv and CREDITS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
