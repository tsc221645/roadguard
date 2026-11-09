"""Evalúa un checkpoint YOLO sobre un split congelado y guarda métricas JSON."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/processed/rdd2022/dataset.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=PROJECT / "artifacts/evaluations")
    args = parser.parse_args()
    model = YOLO(str(args.model.resolve()))
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers,
        project=str(args.project.resolve()), name=args.name, plots=True,
    )
    report = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "results": {key: float(value) for key, value in metrics.results_dict.items()},
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class_map50_95": [float(value) for value in metrics.box.maps],
    }
    output = args.project / args.name / "metrics.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
