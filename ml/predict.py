"""Ejecuta inferencia de RoadGuard sobre una imagen, carpeta, video o webcam."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Imagen, carpeta, video o índice de webcam")
    parser.add_argument("--model", type=Path,
                        default=PROJECT / "artifacts/runs/model_b/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.10,
                        help="Confianza mínima; 0.10 fue seleccionada en validación")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=PROJECT / "artifacts/predictions")
    parser.add_argument("--name", default="latest")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Guarda detecciones estructuradas en detections.json")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf debe estar entre 0 y 1")

    model = YOLO(str(args.model.resolve()))
    results = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, project=str(args.output), name=args.name,
                            exist_ok=True, save=True, save_txt=args.save_txt,
                            save_conf=args.save_conf, verbose=False)
    output_dir = args.output / args.name
    if args.report:
        detections = []
        for result in results:
            boxes = result.boxes
            rows = []
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                for coordinates, confidence, class_id in zip(xyxy, confidences, classes):
                    class_id = int(class_id)
                    rows.append({
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in coordinates],
                    })
            detections.append({"source": str(result.path), "detections": rows})
        (output_dir / "detections.json").write_text(
            json.dumps({"model": str(args.model.resolve()), "conf": args.conf,
                        "items": detections}, indent=2) + "\n",
            encoding="utf-8")
    print(f"Procesados: {len(results)}")
    print(f"Resultados: {output_dir}")


if __name__ == "__main__":
    main()
