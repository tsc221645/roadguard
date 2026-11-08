"""Inventario recursivo RDD2022. Ejecutar: python -m ml.inspect_rdd --help."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from ml.dataset import IMAGE_SUFFIXES, image_size, parse_voc, sha256, write_csv, write_json

COUNTRIES = {"japan": "Japan", "india": "India", "czech": "Czech",
             "norway": "Norway", "united_states": "United_States", "china": "China"}
FIELDS = ["sample_id", "source", "country", "camera_domain", "image_path", "xml_path",
          "image_bytes", "sha256", "xml_sha256", "width", "height", "status", "error",
          "boxes", "ignored_classes"]


def origin(path):
    lower = path.as_posix().lower()
    country = next((value for key, value in COUNTRIES.items() if key in lower), "unknown")
    # Solo inferencias a partir de nombres de archivos; no inventar metadata de cámara.
    domain = "drone" if "drone" in lower else "motorbike" if "motorbike" in lower else "unknown"
    return country, domain


def inspect(root, output):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"No existe la raíz de datos: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix.lower() in IMAGE_SUFFIXES | {".xml"})
    if not files:
        raise ValueError("No se encontraron imágenes/XML; descomprime el dataset primero")
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    xmls = [p for p in files if p.suffix.lower() == ".xml"]
    image_index, xml_index = defaultdict(list), defaultdict(list)
    for paths, index in ((images, image_index), (xmls, xml_index)):
        for path in paths:
            index[path.stem.casefold()].append(path)
    rows, used = [], set()
    for number, image in enumerate(images, 1):
        relative = image.relative_to(root)
        country, domain = origin(relative)
        row = dict.fromkeys(FIELDS, "")
        row.update(sample_id=sha256_string(relative.as_posix())[:24], source="RDD2022",
                   country=country, camera_domain=domain, image_path=relative.as_posix(),
                   image_bytes=image.stat().st_size, sha256=sha256(image),
                   boxes="[]", ignored_classes="[]")
        candidates = xml_index[image.stem.casefold()]
        # El stem de RDD normalmente incluye país; si se repite no adivinar asociaciones.
        if len(candidates) == 1 and len(image_index[image.stem.casefold()]) == 1:
            xml = candidates[0]
            used.add(xml)
            row.update(xml_path=xml.relative_to(root).as_posix(), xml_sha256=sha256(xml))
        else:
            xml = None
        try:
            size = image_size(image)
            row.update(width=size[0], height=size[1])
            if len(candidates) > 1 or len(image_index[image.stem.casefold()]) > 1:
                row.update(status="ambiguous_pair", error="Stem repetido; asociación requiere revisión")
            elif xml is None:
                row.update(status="missing_xml")
            else:
                boxes, ignored = parse_voc(xml, size)
                row.update(boxes=json.dumps(boxes), ignored_classes=json.dumps(ignored),
                           status="ignored_only" if ignored and not boxes else "ready")
        except (ValueError, OSError, SyntaxError) as error:
            row.update(status="invalid", error=str(error))
        rows.append(row)
        if number % 1000 == 0:
            print(f"Inventariadas {number}/{len(images)} imágenes", flush=True)
    for xml in xmls:
        if xml not in used:
            row = dict.fromkeys(FIELDS, "")
            country, domain = origin(xml.relative_to(root))
            row.update(sample_id=sha256_string(xml.relative_to(root).as_posix())[:24],
                       source="RDD2022", country=country, camera_domain=domain,
                       xml_path=xml.relative_to(root).as_posix(), xml_sha256=sha256(xml),
                       status="unpaired_xml", boxes="[]", ignored_classes="[]")
            rows.append(row)
    output = Path(output)
    write_csv(output, rows, FIELDS)
    hashes = Counter(row["sha256"] for row in rows if row["sha256"])
    report = {"images": len(images), "xml_files": len(xmls),
              "status_counts": dict(Counter(r["status"] for r in rows)),
              "duplicate_image_copies": sum(n - 1 for n in hashes.values()),
              "country_images": dict(Counter(r["country"] for r in rows if r["image_path"])),
              "inventory_sha256": sha256(output),
              "coordinate_convention": "continuous_0_W_0_H_no_clipping"}
    write_json(output.with_suffix(".summary.json"), report)
    return report


def sha256_string(value):
    import hashlib
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/rdd_inventory.csv"))
    args = parser.parse_args()
    print(json.dumps(inspect(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
