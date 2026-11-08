"""Verifica integridad de dataset congelado y fugas entre splits."""
import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from ml.dataset import classes, image_size, read_csv, read_labels, safe_path, sha256


def validate(root):
    root = Path(root)
    report = json.loads((root / "manifests/split_report.json").read_text())
    config = yaml.safe_load((root / "dataset.yaml").read_text())
    if config["names"] != classes() or report["classes"] != {str(k): v for k, v in classes().items()}:
        raise ValueError("Clases del dataset no coinciden con configuración")
    hashes, groups, ids, expected_images, expected_labels = {}, {}, set(), set(), set()
    counts = {}
    for split in ("train", "val", "test"):
        path = root / "manifests" / f"{split}.csv"
        if sha256(path) != report["splits"][split]["manifest_sha256"]:
            raise ValueError(f"Manifiesto congelado modificado: {split}")
        rows = read_csv(path)
        if not rows or len(rows) != report["splits"][split]["images"]:
            raise ValueError(f"Split vacío o conteo inconsistente: {split}")
        class_counts = Counter()
        for row in rows:
            if row["split"] != split or row["sample_id"] in ids:
                raise ValueError("ID repetido o split inconsistente")
            ids.add(row["sample_id"])
            if row["sha256"] in hashes:
                raise ValueError("Hash repetido en dataset")
            hashes[row["sha256"]] = split
            group = row["group_id"]
            if not group or (group in groups and groups[group] != split):
                raise ValueError("Grupo vacío o cruzado entre splits")
            groups[group] = split
            if not row["image_path"].startswith(f"images/{split}/") or not row["label_path"].startswith(f"labels/{split}/"):
                raise ValueError("Ruta no corresponde al split")
            image, label = safe_path(root, row["image_path"]), safe_path(root, row["label_path"])
            expected_images.add(image)
            expected_labels.add(label)
            if sha256(image) != row["sha256"] or sha256(label) != row["label_sha256"]:
                raise ValueError("Contenido modificado desde congelamiento")
            if image_size(image) != (int(row["width"]), int(row["height"])):
                raise ValueError("Dimensiones no coinciden")
            boxes = read_labels(label)
            for index, name in classes().items():
                count = sum(b[0] == index for b in boxes)
                if count != int(row[f"count_{name}"]):
                    raise ValueError("Conteo de clases distinto del manifiesto")
                class_counts[name] += count
        if dict(class_counts) != report["splits"][split]["class_instances"]:
            raise ValueError("Conteo de clases distinto del reporte")
        counts[split] = len(rows)
    actual_images = {p.resolve() for p in (root / "images").rglob("*") if p.is_file()}
    actual_labels = {p.resolve() for p in (root / "labels").rglob("*") if p.is_file()}
    if actual_images != expected_images or actual_labels != expected_labels:
        raise ValueError("Imágenes/etiquetas extras o faltantes fuera del manifiesto")
    return {"valid": True, "images": counts, "groups": len(groups), "duplicate_hashes": 0,
            "cross_split_groups": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/rdd2022"))
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), indent=2))


if __name__ == "__main__":
    main()
