"""EDA de inventario y overlays YOLO, antes de entrenar."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ml.dataset import classes, read_csv, read_labels, safe_path, sha256, write_json


def describe(values):
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=float)
    return {"n": len(values), "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "p50": float(np.median(array)),
            "p95": float(np.percentile(array, 95))}


def analyze(root, inventory, converted, output, per_class=20, seed=42):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Reporte existente: {output}")
    rows = read_csv(inventory)
    if not rows:
        raise ValueError("Inventario vacío")
    output.mkdir(parents=True)
    names = classes()
    class_counts, countries, ignored = Counter(), Counter(), Counter()
    objects, widths, heights, aspects = [], [], [], []
    areas = {name: [] for name in names.values()}
    by_country = {}
    for row in rows:
        if not row["image_path"]:
            continue
        countries[row["country"]] += 1
        if row["width"] and row["height"]:
            widths.append(int(row["width"]))
            heights.append(int(row["height"]))
            aspects.append(int(row["width"]) / int(row["height"]))
        if row["status"] not in {"ready", "ignored_only"}:
            continue
        boxes = json.loads(row["boxes"])
        objects.append(len(boxes))
        ignored.update(json.loads(row["ignored_classes"]))
        for box in boxes:
            name = names[box[0]]
            class_counts[name] += 1
            by_country.setdefault(row["country"], Counter())[name] += 1
            areas[name].append(box[3] * box[4])
    hashes = Counter(r["sha256"] for r in rows if r["sha256"])
    report = {"inventory_sha256": sha256(inventory),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "country_images": dict(countries), "class_instances": dict(class_counts),
              "class_percent": {name: 100 * class_counts[name] / max(sum(class_counts.values()), 1)
                                for name in names.values()},
              "classes_by_country": {country: dict(counts) for country, counts in by_country.items()},
              "objects_per_annotated_image": dict(Counter(objects)),
              "ignored_classes": dict(ignored), "duplicate_copies": sum(n - 1 for n in hashes.values()),
              "width": describe(widths), "height": describe(heights), "aspect_ratio": describe(aspects),
              "relative_bbox_area": {name: describe(values) for name, values in areas.items()},
              "camera_domain_images": dict(Counter(r["camera_domain"] for r in rows if r["image_path"])),
              "limitations": ["País/cámara inferidos de rutas, no geolocalización real",
                              "Iluminación y superficie requieren revisión visual/manual",
                              "Duplicados perceptuales y secuencias no se certifican con SHA-256"]}
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(list(names.values()), [class_counts[name] for name in names.values()])
    axes[0].set_title("Instancias por clase")
    axes[1].bar(list(countries), list(countries.values()))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_title("Imágenes por país")
    axes[2].hist([value for values in areas.values() for value in values], bins=30)
    axes[2].set_title("Área relativa de cajas")
    figure.tight_layout()
    figure.savefig(output / "distributions.png", dpi=150)
    plt.close(figure)
    converted = Path(converted)
    candidates = read_csv(converted / "converted.csv")
    rng = random.Random(seed)
    report["overlays"] = {}
    for index, name in names.items():
        eligible = [r for r in candidates if int(r[f"count_{name}"]) > 0]
        rng.shuffle(eligible)
        selected = eligible[:per_class]
        report["overlays"][name] = [row["sample_id"] for row in selected]
        if not selected:
            continue
        sheet = Image.new("RGB", (4 * 320, ((len(selected) + 3) // 4) * 264), "white")
        for position, row in enumerate(selected):
            with Image.open(safe_path(root, row["image_path"])) as source:
                image = source.convert("RGB")
            width, height = image.size
            draw = ImageDraw.Draw(image)
            for cls, x, y, w, h in read_labels(safe_path(converted, row["label_path"])):
                color = ("red", "blue", "lime", "orange")[cls]
                draw.rectangle(((x - w / 2) * width, (y - h / 2) * height,
                                (x + w / 2) * width, (y + h / 2) * height), outline=color, width=3)
                draw.text(((x - w / 2) * width, (y - h / 2) * height), names[cls], fill=color)
            tile = ImageOps.pad(image, (320, 240), color="white")
            x0, y0 = (position % 4) * 320, (position // 4) * 264
            sheet.paste(tile, (x0, y0))
            ImageDraw.Draw(sheet).text((x0 + 4, y0 + 242), row["sample_id"][:16], fill="black")
        sheet.save(output / f"overlays_{name}.jpg")
    write_json(output / "eda.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, default=Path("data/manifests/rdd_inventory.csv"))
    parser.add_argument("--converted", type=Path, default=Path("data/intermediate/rdd2022"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/eda"))
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("per-class debe ser positivo")
    report = analyze(args.root, args.inventory, args.converted, args.output, args.per_class)
    print(json.dumps({"class_instances": report["class_instances"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
