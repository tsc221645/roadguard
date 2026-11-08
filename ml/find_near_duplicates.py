"""Candidatos de similitud perceptual para auditar fugas; no prueba identidad.

pHash DCT de 63 bits (DC excluido) y búsqueda exacta de vecinos por Hamming
mediante BK-tree. No confundir estos candidatos con secuencias verificadas.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn

from ml.dataset import read_csv, safe_path, sha256, write_csv, write_json


def perceptual_hash(path):
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    values = dctn(np.asarray(gray, dtype=float), type=2, norm="ortho")[:8, :8].ravel()[1:]
    result = 0
    for value in values > np.median(values):
        result = (result << 1) | int(value)
    return result


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, identifier):
        if self.root is None:
            self.root = [value, [identifier], {}]
            return
        node = self.root
        while True:
            distance = (value ^ node[0]).bit_count()
            if distance == 0:
                node[1].append(identifier)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [identifier], {}]
                return
            node = node[2][distance]

    def neighbors(self, value, radius):
        pending = [self.root] if self.root is not None else []
        result = []
        while pending:
            current, identifiers, children = pending.pop()
            distance = (value ^ current).bit_count()
            if distance <= radius:
                result.extend((identifier, distance) for identifier in identifiers)
            pending.extend(node for edge, node in children.items()
                           if distance - radius <= edge <= distance + radius)
        return result


def find(root, manifest, output, radius=6):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Reporte existente: {output}")
    if not 0 <= radius <= 16:
        raise ValueError("Radio Hamming debe estar entre 0 y 16")
    rows = read_csv(manifest)
    tree, hashes, edges = BKTree(), [], []
    for index, row in enumerate(rows):
        image = safe_path(root, row["image_path"])
        if sha256(image) != row["sha256"]:
            raise ValueError("Imagen modificada desde inventario")
        value = perceptual_hash(image)
        for other, distance in tree.neighbors(value, radius):
            edges.append({"sample_a": other, "sample_b": row["sample_id"], "distance": distance})
        tree.add(value, row["sample_id"])
        hashes.append({"sample_id": row["sample_id"], "phash": f"{value:016x}"})
        if (index + 1) % 1000 == 0:
            print(f"pHash: {index + 1}/{len(rows)}", flush=True)
    output.mkdir(parents=True)
    write_csv(output / "candidates.csv", edges, ["sample_a", "sample_b", "distance"])
    write_csv(output / "hashes.csv", hashes, ["sample_id", "phash"])
    report = {"images": len(rows), "candidate_pairs": len(edges), "hamming_radius": radius,
              "distance_counts": dict(Counter(edge["distance"] for edge in edges)),
              "manifest_sha256": sha256(manifest),
              "interpretation": "Candidatos para revisión; pueden incluir falsos positivos y omitir vecinos reales"}
    write_json(output / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=6)
    args = parser.parse_args()
    import json
    print(json.dumps(find(args.root, args.manifest, args.output, args.radius), indent=2))


if __name__ == "__main__":
    main()
