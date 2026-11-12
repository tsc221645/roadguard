"""Cuenta TP/FP/FN por clase en un split YOLO con confianza e IoU fijos."""
import argparse
import json
import statistics
import time
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("D00", "D10", "D20", "D40")


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def truth_boxes(label_path, width, height):
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, w, h = map(float, line.split()[:5])
        labels.append((int(class_id), ((xc - w / 2) * width, (yc - h / 2) * height,
                                        (xc + w / 2) * width, (yc + h / 2) * height)))
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if not 0 < args.conf < 1 or not 0 < args.iou <= 1:
        parser.error("conf e iou deben estar en (0, 1]")

    image_dir = ROOT / "data/processed/rdd2022/images" / args.split
    label_dir = ROOT / "data/processed/rdd2022/labels" / args.split
    model = YOLO(str(args.model.resolve()))
    counts = {name: {"tp": 0, "fp": 0, "fn": 0, "ground_truth": 0} for name in NAMES}
    per_image = []
    inference_ms = []
    start = time.perf_counter()
    images = 0
    for result in model.predict(source=str(image_dir), conf=args.conf, imgsz=args.imgsz,
                                device=args.device, stream=True, verbose=False, save=False):
        images += 1
        height, width = result.orig_shape
        truths = truth_boxes(label_dir / (Path(result.path).stem + ".txt"), width, height)
        before = {key: sum(row[key] for row in counts.values()) for key in ("tp", "fp", "fn")}
        for class_id, _ in truths:
            counts[NAMES[class_id]]["ground_truth"] += 1
        matched = set()
        boxes = result.boxes
        detections = sorted(zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
                                boxes.xyxy.cpu().tolist()), key=lambda row: row[1], reverse=True)
        for class_id, _, box in detections:
            class_id = int(class_id)
            candidates = [(overlap(box, ground_box), index)
                          for index, (ground_class, ground_box) in enumerate(truths)
                          if ground_class == class_id and index not in matched]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= args.iou:
                matched.add(best_index)
                counts[NAMES[class_id]]["tp"] += 1
            else:
                counts[NAMES[class_id]]["fp"] += 1
        for index, (class_id, _) in enumerate(truths):
            if index not in matched:
                counts[NAMES[class_id]]["fn"] += 1
        per_image.append({"image": Path(result.path).name, **{
            key: sum(row[key] for row in counts.values()) - before[key]
            for key in ("tp", "fp", "fn")}})
        inference_ms.append(float(result.speed.get("inference", 0)))
        if images % 250 == 0:
            print(f"{args.name}: {images} imágenes procesadas", flush=True)

    totals = {key: sum(row[key] for row in counts.values())
              for key in ("tp", "fp", "fn", "ground_truth")}
    for row in list(counts.values()) + [totals]:
        row["precision"] = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0
        row["recall"] = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0
        row["f1"] = 2 * row["tp"] / (2 * row["tp"] + row["fp"] + row["fn"]) if 2 * row["tp"] + row["fp"] + row["fn"] else 0
    report = {"model": str(args.model.resolve()), "split": args.split,
              "confidence": args.conf, "matching_iou": args.iou, "imgsz": args.imgsz,
              "images": images, "per_class": counts, "total": totals,
              "per_image": per_image,
              "wall_seconds": round(time.perf_counter() - start, 2),
              "inference_ms_median": round(statistics.median(inference_ms), 2),
              "inference_ms_p95": round(sorted(inference_ms)[min(len(inference_ms) - 1,
                                                                 int(0.95 * len(inference_ms)))], 2)}
    output = ROOT / "artifacts/evaluations" / args.name / "operational.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
