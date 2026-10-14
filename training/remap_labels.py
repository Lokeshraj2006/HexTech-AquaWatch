"""Turn any YOLO-format swimmer / drowning dataset into OUR 5 body-state classes.

Public datasets use their own labels ("swimming", "drowning", "out of water", "person" ...). We do NOT
train a swimming-vs-drowning classifier: a "drowning" box only tells us what the BODY looks like in that
frame (usually upright or head-only). The distress decision is made later, over time (distress.py).

Our classes (fixed for the whole team, see training/README.md):
    0 swimmer_horizontal   body flat, swimming
    1 swimmer_vertical     body upright in the water (treading OR distress: the model must not decide which)
    2 head_only            only the head is above the water
    3 underwater           body visible below the surface
    4 out_of_water         on the deck / outside the pool (ignored by the pipeline)

Usage:
    python training/remap_labels.py --src path/to/dataset --map training/label_map.example.yaml \\
                                    --out datasets/swimmers [--split 0.8 0.1 0.1]

`--src` is a YOLO dataset folder with data.yaml (class names) and train/valid/test (or val) sub-folders,
each with images/ and labels/. Boxes whose class maps to "drop" are removed. With --split, all images are
pooled and re-split by image (use it when the dataset has no validation / test split).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

OUR_CLASSES = ["swimmer_horizontal", "swimmer_vertical", "head_only", "underwater", "out_of_water"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(src: Path) -> list[str]:
    """Class names from the source data.yaml (list or {id: name} dict)."""
    data = yaml.safe_load(open(src / "data.yaml", encoding="utf-8"))
    names = data["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def build_lookup(src_names: list[str], mapping: dict) -> dict[int, int | None]:
    """Source class id -> our class id (None = drop). Names are matched case-insensitively."""
    norm = {str(k).strip().lower(): v for k, v in mapping.items()}
    lookup = {}
    for i, name in enumerate(src_names):
        target = norm.get(name.strip().lower(), "drop")
        if target == "drop" or target is None:
            lookup[i] = None
        elif target in OUR_CLASSES:
            lookup[i] = OUR_CLASSES.index(target)
        else:
            raise SystemExit(f"label map: '{name}' -> '{target}' is not one of {OUR_CLASSES} or 'drop'")
    return lookup


def find_pairs(src: Path) -> dict[str, list[tuple[Path, Path]]]:
    """{split: [(image, label_file)]} for the splits that exist."""
    pairs = {}
    for split in ("train", "valid", "val", "test"):
        img_dir = src / split / "images"
        if not img_dir.is_dir():
            continue
        lab_dir = src / split / "labels"
        items = [(p, lab_dir / (p.stem + ".txt")) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
        pairs["val" if split == "valid" else split] = items
    return pairs


def remap_file(label: Path, lookup: dict, counts_in: Counter, counts_out: Counter) -> list[str]:
    """Remapped lines of one YOLO label file (boxes of dropped classes removed)."""
    out = []
    if not label.exists():
        return out
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(float(parts[0]))
        counts_in[cid] += 1
        new = lookup.get(cid)
        if new is not None:
            counts_out[new] += 1
            out.append(" ".join([str(new)] + parts[1:]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--map", required=True, help="YAML: source class name -> one of our classes, or drop")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"),
                    help="pool all images and re-split by these fractions")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    src_names = load_names(src)
    lookup = build_lookup(src_names, yaml.safe_load(open(args.map, encoding="utf-8")) or {})
    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"no train/valid/test folders with images/ found in {src}")
    if args.split:
        everything = [p for items in pairs.values() for p in items]
        random.Random(args.seed).shuffle(everything)
        a, b = int(len(everything) * args.split[0]), int(len(everything) * (args.split[0] + args.split[1]))
        pairs = {"train": everything[:a], "val": everything[a:b], "test": everything[b:]}
    counts_in, counts_out, kept = Counter(), Counter(), Counter()
    for split, items in pairs.items():
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            lines = remap_file(lab, lookup, counts_in, counts_out)
            shutil.copy2(img, out / split / "images" / img.name)
            (out / split / "labels" / (img.stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
            kept[split] += 1
    data = {"path": str(out.resolve()), "train": "train/images", "val": "val/images",
            "names": {i: n for i, n in enumerate(OUR_CLASSES)}}
    if "test" in pairs:
        data["test"] = "test/images"
    yaml.safe_dump(data, open(out / "data.yaml", "w", encoding="utf-8"), sort_keys=False)
    print("Images per split:", dict(kept))
    print("Boxes in  (source classes):", {src_names[k]: v for k, v in sorted(counts_in.items())})
    print("Boxes out (our classes):   ", {OUR_CLASSES[k]: v for k, v in sorted(counts_out.items())})
    missing = [c for i, c in enumerate(OUR_CLASSES) if counts_out[i] == 0]
    if missing:
        print("WARNING: no boxes for", missing, "- the model cannot learn these; label more or drop them.")
    print(f"Wrote {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
