"""Busca un umbral de confianza con precision/recall del split de validación."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/processed/rdd2022/dataset.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/thresholds")
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.thresholds):
        parser.error("Todos los umbrales deben estar entre 0 y 1")
    model = YOLO(str(args.model.resolve()))
    rows = []
    validation_output = args.output / args.name / "validation_runs"
    validation_output.mkdir(parents=True, exist_ok=True)
    for threshold in args.thresholds:
        metrics = model.val(data=str(args.data.resolve()), split="val", conf=threshold,
                            imgsz=args.imgsz, batch=args.batch, device=args.device,
                            workers=args.workers, plots=False, verbose=False,
                            project=str(validation_output), name=f"conf_{threshold:.2f}",
                            exist_ok=True)
        values = metrics.results_dict
        precision = float(values["metrics/precision(B)"])
        recall = float(values["metrics/recall(B)"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": threshold, "precision": precision, "recall": recall,
                     "f1": f1, "map50": float(values["metrics/mAP50(B)"]),
                     "map50_95": float(values["metrics/mAP50-95(B)"])})
        print(rows[-1])
    best = max(rows, key=lambda row: row["f1"])
    report = {"model": str(args.model.resolve()), "split": "val", "rows": rows,
              "selected": best, "selection_metric": "f1"}
    output = args.output / args.name
    output.mkdir(parents=True, exist_ok=True)
    (output / "thresholds.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
