"""Entrenamiento reproducible de RoadGuard con Ultralytics.

Ejemplos (usar el entorno Conda dm1):
  python -m ml.train --name model_a --epochs 25
  python -m ml.train --name model_b --epochs 25 --lr0 0.0005 --degrees 3
  python -m ml.train --name model_c --epochs 50 --lr0 0.0005 --degrees 3
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                        choices=("model_a", "model_b", "model_c", "smoke_a", "smoke_b"))
    parser.add_argument("--data", type=Path, default=PROJECT / "data/processed/rdd2022/dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--project", type=Path, default=PROJECT / "artifacts/runs")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction debe estar en (0, 1]")
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs y batch deben ser positivos")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        lr0=args.lr0,
        degrees=args.degrees,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
