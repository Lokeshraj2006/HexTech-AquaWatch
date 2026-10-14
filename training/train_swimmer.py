"""Fine-tune YOLO11n into a swimmer body-state detector (5 classes), evaluate it, and export it.

    python training/train_swimmer.py --data datasets/swimmers/data.yaml --epochs 60 --device 0
    python training/train_swimmer.py --data datasets/swimmers/data.yaml --evaluate-only \\
                                     --weights training/runs/swimmer/weights/best.pt

Run it on Colab with a T4 (see training/README.md). At the end it prints mAP per class on the test split
(or val), writes training/runs/<name>/eval.json, exports ONNX, and prints the exact lines to paste into
scenarios/pool.yaml so the pipeline uses the new model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def evaluate(weights: str, data: str, device: str, out_json: Path, imgsz: int = 640) -> dict:
    """mAP50 / mAP50-95 overall and per class on the test split (val if there is no test split)."""
    import yaml
    from ultralytics import YOLO
    split = "test" if "test" in (yaml.safe_load(open(data, encoding="utf-8")) or {}) else "val"
    metrics = YOLO(weights).val(data=data, split=split, device=device, imgsz=imgsz, verbose=False)
    names = metrics.names
    per_class = {names[int(c)]: {"mAP50": round(float(metrics.box.ap50[i]), 3), "mAP50_95": round(float(metrics.box.ap[i]), 3)}
                 for i, c in enumerate(metrics.box.ap_class_index)}
    result = {"split": split, "weights": weights, "mAP50": round(float(metrics.box.map50), 3),
              "mAP50_95": round(float(metrics.box.map), 3), "precision": round(float(metrics.box.mp), 3),
              "recall": round(float(metrics.box.mr), 3), "per_class": per_class}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n{split} split: mAP50 {result['mAP50']}  mAP50-95 {result['mAP50_95']}  "
          f"precision {result['precision']}  recall {result['recall']}")
    for name, m in per_class.items():
        print(f"  {name:<20} mAP50 {m['mAP50']:.3f}  mAP50-95 {m['mAP50_95']:.3f}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="data.yaml made by remap_labels.py")
    ap.add_argument("--model", default="yolo11n.pt", help="start from these pretrained weights")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--name", default="swimmer")
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--weights", default=None, help="with --evaluate-only: the model to evaluate")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="sanity check on CPU: 1 epoch on 5%% of the train set at 320 px, val only, no export")
    args = ap.parse_args(argv)
    if args.quick:
        args.epochs, args.imgsz, args.batch, args.no_export = 1, 320, 16, True
        args.name = args.name if args.name != "swimmer" else "swimmer_quick"
    project = HERE / "runs"
    run_dir = project / args.name
    if args.evaluate_only:
        weights = args.weights or str(run_dir / "weights" / "best.pt")
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        # Augmentation tuned for pools: colour shifts (water / lighting), flips; no vertical flip
        # (an upside-down swimmer would teach the wrong posture).
        model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
                    project=str(project), name=args.name, exist_ok=True, patience=20, hsv_h=0.02, hsv_s=0.6,
                    hsv_v=0.4, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=10, plots=True,
                    fraction=0.05 if args.quick else 1.0, workers=0 if args.quick else 8)
        weights = str(run_dir / "weights" / "best.pt")
    evaluate(weights, args.data, args.device, run_dir / "eval.json", args.imgsz)
    if not args.no_export:
        from ultralytics import YOLO
        onnx = YOLO(weights).export(format="onnx", imgsz=args.imgsz, dynamic=False)
        print(f"Exported ONNX: {onnx}")
    try:                                   # the path the preset needs: relative to the project folder
        rel = Path(weights).resolve().relative_to(HERE.parent).as_posix()
    except ValueError:
        rel = Path(weights).as_posix()
    print("\nUse it in the pipeline with the pool_trained preset:")
    print("  python run.py --video pool.mp4 --zones pool_zones.json --scenario pool_trained")
    if rel != "training/runs/swimmer/weights/best.pt":
        print(f"It expects training/runs/swimmer/weights/best.pt; for this model set in scenarios/pool_trained.yaml\n"
              f"  model:\n    weights: {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
