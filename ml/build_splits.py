"""División por grupos explícitos con congelamiento y detección de duplicados.

CSV --groups: sample_id,group_id. No inventa secuencias a partir de un contador.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import yaml

from ml.dataset import (classes, read_csv, safe_path, sha256, write_csv, write_json)


def assign_groups(rows, groups, similarity_links=None):
    mapping = {}
    for row in read_csv(groups):
        if not row.get("group_id", "").strip() or row["sample_id"] in mapping:
            raise ValueError("Cada sample_id requiere exactamente un group_id no vacío")
        mapping[row["sample_id"]] = row["group_id"].strip()
    missing = {r["sample_id"] for r in rows} - mapping.keys()
    if missing:
        raise ValueError(f"Faltan grupos para {len(missing)} muestras")
    # Unir grupos conectados por copias exactas ANTES de eliminar duplicados.
    parent = {value: value for value in mapping.values()}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    selected = {row["sample_id"] for row in rows}
    if similarity_links is not None:
        for edge in read_csv(similarity_links):
            # Un filtro explícito de países puede dejar enlaces fuera del subconjunto.
            if edge["sample_a"] in selected and edge["sample_b"] in selected:
                first = find(mapping[edge["sample_a"]])
                parent[find(mapping[edge["sample_b"]])] = first
    unique, duplicate_count = [], 0
    for copies in by_hash.values():
        if len({r["label_sha256"] for r in copies}) != 1:
            raise ValueError("Imagen duplicada con anotaciones contradictorias; revisar antes de dividir")
        first = find(mapping[copies[0]["sample_id"]])
        for row in copies[1:]:
            parent[find(mapping[row["sample_id"]])] = first
        unique.append(copies[0].copy())
        duplicate_count += len(copies) - 1
    components = defaultdict(list)
    for group in parent:
        components[find(group)].append(group)
    ids = {key: hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()[:24]
           for key, values in components.items()}
    for row in unique:
        row["original_group_id"] = mapping[row["sample_id"]]
        row["group_id"] = ids[find(row["original_group_id"])]
    return unique, duplicate_count


def partition(rows, ratios, seed):
    """Aproximar proporciones de imágenes, clases y países sin romper grupos."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    if len(grouped) < 3:
        raise ValueError("Se requieren al menos tres grupos independientes")
    def features(members):
        counts = Counter({"images": len(members)})
        for row in members:
            counts[f"country:{row['country']}"] += 1
            for name in classes().values():
                counts[f"class:{name}"] += int(row[f"count_{name}"])
        return counts
    total = features(rows)
    items = list(grouped.items())
    random.Random(seed).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    result = {name: [] for name in ratios}
    current = {name: Counter() for name in ratios}
    for index, (_, members) in enumerate(items):
        vector = features(members)
        empty = [name for name in ratios if not result[name]]
        choices = empty if len(items) - index == len(empty) else list(ratios)
        def score(candidate):
            # Cambio de error cuadrático; normalizar por frecuencia total, no por split.
            return sum(((current[candidate][key] + count - ratios[candidate] * total[key]) ** 2
                        - (current[candidate][key] - ratios[candidate] * total[key]) ** 2)
                       / max(total[key], 1) ** 2 for key, count in vector.items())
        chosen = min(choices, key=score)
        result[chosen].extend(members)
        current[chosen].update(vector)
    return result


def build(root, converted, groups, output, seed=42, ratios=(.7, .15, .15), countries=None,
          similarity_links=None):
    if len(ratios) != 3 or any(r <= 0 for r in ratios) or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Tres proporciones positivas deben sumar 1")
    output, converted = Path(output), Path(converted)
    if output.exists():
        raise ValueError(f"Dataset congelado/salida existente: {output}")
    rows = read_csv(converted / "converted.csv")
    if countries:
        rows = [row for row in rows if row["country"] in countries]
    if not rows:
        raise ValueError("No hay muestras elegibles")
    rows, duplicates = assign_groups(rows, groups, similarity_links)
    splits = partition(rows, dict(zip(("train", "val", "test"), ratios)), seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".split-", dir=output.parent))
    try:
        report = {"seed": seed, "requested_ratios": ratios, "duplicate_copies_removed": duplicates,
                  "group_map_sha256": sha256(groups),
                  "similarity_links_sha256": sha256(similarity_links) if similarity_links else None,
                  "converted_manifest_sha256": sha256(converted / "converted.csv"),
                  "classes": classes(), "splits": {},
                  "grouping_warning": "La independencia real depende de la justificación del CSV de grupos"}
        for split, members in splits.items():
            for row in members:
                image = safe_path(root, row["image_path"])
                label = safe_path(converted, row["label_path"])
                if sha256(image) != row["sha256"] or sha256(label) != row["label_sha256"]:
                    raise ValueError(f"Fuente modificada: {row['sample_id']}")
                row["source_image_path"] = row["image_path"]
                row["image_path"] = f"images/{split}/{row['sample_id']}{image.suffix.lower()}"
                row["label_path"] = f"labels/{split}/{row['sample_id']}.txt"
                row["split"] = split
                for src, relative in ((image, row["image_path"]), (label, row["label_path"])):
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, target)
            write_csv(stage / "manifests" / f"{split}.csv", members, list(members[0]))
            counts = {name: sum(int(row[f"count_{name}"]) for row in members)
                      for name in classes().values()}
            report["splits"][split] = {"images": len(members), "groups": len({r['group_id'] for r in members}),
                                       "class_instances": counts,
                                       "missing_classes": [name for name, count in counts.items() if not count],
                                       "manifest_sha256": sha256(stage / "manifests" / f"{split}.csv")}
        (stage / "dataset.yaml").write_text(yaml.safe_dump({"path": str(output.resolve()),
               "train": "images/train", "val": "images/val", "test": "images/test", "names": classes()},
               sort_keys=False))
        write_json(stage / "manifests/split_report.json", report)
        # Comprobar antes de publicar la carpeta final, sin aceptar artefactos incompletos.
        from ml.validate_dataset import validate
        validate(stage)
        stage.rename(output)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--converted", type=Path, default=Path("data/intermediate/rdd2022"))
    parser.add_argument("--groups", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/rdd2022"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", type=float, nargs=3, default=(.7, .15, .15))
    parser.add_argument("--countries", nargs="+", help="Filtro explícito previo al split; registrar justificación")
    parser.add_argument("--similarity-links", type=Path,
                        help="CSV sample_a,sample_b revisado: unir grupos conectados ANTES de dividir")
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.converted, args.groups, args.output,
                           args.seed, args.ratios, args.countries, args.similarity_links), indent=2))


if __name__ == "__main__":
    main()
