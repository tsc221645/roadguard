"""Conversión auditada; no modifica imágenes/XML originales."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import tempfile

from ml.dataset import (classes, image_size, label_text, parse_voc, read_csv,
                        safe_path, sha256, write_csv, write_json)


def convert(root, inventory, output, include_negatives=False):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Salida existente, no se sobrescribe: {output}")
    rows = read_csv(inventory)
    if not rows:
        raise ValueError("Inventario vacío")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".conversion-", dir=output.parent))
    accepted, excluded = [], []
    try:
        for row in rows:
            reason = row["status"] if row["status"] != "ready" else ""
            if not reason and not json.loads(row["boxes"]) and not include_negatives:
                reason = "empty_annotation_requires_opt_in"
            if reason:
                excluded.append({"sample_id": row["sample_id"], "reason": reason})
                continue
            image = safe_path(root, row["image_path"])
            xml = safe_path(root, row["xml_path"])
            if sha256(image) != row["sha256"] or sha256(xml) != row["xml_sha256"]:
                raise ValueError(f"Archivo cambió desde inventario: {row['sample_id']}")
            boxes, ignored = parse_voc(xml, image_size(image))
            if boxes != json.loads(row["boxes"]) or ignored != json.loads(row["ignored_classes"]):
                raise ValueError("Anotaciones del inventario no coinciden con XML")
            target = stage / "labels" / f"{row['sample_id']}.txt"
            target.parent.mkdir(exist_ok=True)
            target.write_text(label_text(boxes))
            row["label_path"] = f"labels/{target.name}"
            row["label_sha256"] = sha256(target)
            for index, name in classes().items():
                row[f"count_{name}"] = sum(b[0] == index for b in boxes)
            accepted.append(row)
        if not accepted:
            raise ValueError("No hay imágenes elegibles después de aplicar políticas")
        write_csv(stage / "converted.csv", accepted, list(accepted[0]))
        write_csv(stage / "exclusions.csv", excluded, ["sample_id", "reason"])
        report = {"accepted": len(accepted), "excluded": len(excluded),
                  "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
                  "inventory_sha256": sha256(inventory), "include_empty_annotations": include_negatives,
                  "ignored_only_policy": "exclude", "clipping": False,
                  "classes": classes()}
        write_json(stage / "conversion_report.json", report)
        stage.rename(output)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, default=Path("data/manifests/rdd_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/intermediate/rdd2022"))
    parser.add_argument("--include-negatives", action="store_true",
                        help="Incluir XML explícitamente vacíos; clases no objetivo solas siguen excluidas")
    args = parser.parse_args()
    print(json.dumps(convert(args.root, args.inventory, args.output, args.include_negatives), indent=2))


if __name__ == "__main__":
    main()
