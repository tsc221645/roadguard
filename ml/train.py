"""Entrenamiento reproducible de RoadGuard con Ultralytics.

Ejemplos (usar el entorno Conda dm1):
  python -m ml.train --name model_a --epochs 25
  python -m ml.train --name model_b --epochs 25 --lr0 0.0005 --degrees 3
  python -m ml.train --name model_c --epochs 50 --lr0 0.0005 --degrees 3
  python -m ml.train --name model_d --model artifacts/runs/model_b/weights/best.pt --epochs 15 --imgsz 768 --optimizer AdamW --lr0 0.0001 --cos-lr --mosaic 0.2 --close-mosaic 5
  python -m ml.train --name model_e --model yolo11s.pt --epochs 30 --batch 8 --optimizer AdamW --lr0 0.001 --cos-lr --cache ram --patience 8
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                        choices=("model_a", "model_b", "model_c", "model_d", "model_e", "smoke_a", "smoke_b"))
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
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", choices=("ram", "disk", "none"), default="none")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--project", type=Path, default=PROJECT / "artifacts/runs")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction debe estar en (0, 1]")
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs y batch deben ser positivos")
    if not 0 <= args.mosaic <= 1 or not 0 <= args.close_mosaic < args.epochs:
        parser.error("mosaic debe estar en [0, 1] y close-mosaic entre 0 y epochs-1")
    if args.patience < 1:
        parser.error("patience debe ser positivo")
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
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        cache=False if args.cache == "none" else args.cache,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=False,
        seed=42,
        deterministic=True,
    )


if __name__ == "__main__":
    main()
