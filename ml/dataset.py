"""Utilidades de datos sin dependencia de GPU ni de Ultralytics."""
import csv
import hashlib
import json
import math
from pathlib import Path

from lxml import etree
from PIL import Image
import yaml

PROJECT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def classes():
    names = yaml.safe_load((PROJECT / "configs/classes.yaml").read_text())["names"]
    if sorted(names) != list(range(len(names))) or len(set(names.values())) != len(names):
        raise ValueError("IDs de clases deben ser consecutivos y nombres únicos")
    return names


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Ruta fuera de raíz: {relative}")
    return path


def image_size(path):
    with Image.open(path) as image:
        image.load()  # verify() por sí solo no detecta todos los JPEG truncados.
        return image.size


def parse_voc(path, size):
    """Convención VOC usada aquí: coordenadas continuas en [0,W]×[0,H].

    No restar 1 ni recortar silenciosamente; auditar overlays antes de entrenar.
    Una anotación incorrecta excluye la imagen completa, no crea negativos falsos.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    tree = etree.parse(str(path), parser)
    if tree.docinfo.doctype:
        raise ValueError("DTD no permitido")
    root = tree.getroot()
    if root.tag != "annotation":
        raise ValueError("Raíz XML distinta de annotation")
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if (width, height) != tuple(size) or width <= 0 or height <= 0:
        raise ValueError(f"Dimensiones XML {(width, height)} != imagen {size}")
    mapping = {name: index for index, name in classes().items()}
    boxes, ignored = [], []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if not name:
            raise ValueError("Objeto sin clase")
        coords = [float(obj.findtext(f"bndbox/{key}", "nan"))
                  for key in ("xmin", "ymin", "xmax", "ymax")]
        x1, y1, x2, y2 = coords
        if (not all(math.isfinite(v) for v in coords)
                or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError(f"Caja inválida: {name} {coords}")
        if name not in mapping:
            ignored.append(name)
            continue
        boxes.append([mapping[name], (x1 + x2) / (2 * width),
                      (y1 + y2) / (2 * height), (x2 - x1) / width,
                      (y2 - y1) / height])
    return boxes, ignored


def label_text(boxes):
    return "".join(f"{int(b[0])} " + " ".join(f"{v:.10f}" for v in b[1:]) + "\n"
                   for b in boxes)


def read_labels(path):
    boxes = []
    for line in Path(path).read_text().splitlines():
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Fila YOLO inválida: {path}")
        cls = int(values[0])
        x, y, width, height = map(float, values[1:])
        if cls not in classes() or not all(math.isfinite(v) for v in (x, y, width, height)):
            raise ValueError(f"Clase/coordenadas inválidas: {path}")
        if (not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1)
                or x - width / 2 < -1e-8 or y - height / 2 < -1e-8
                or x + width / 2 > 1 + 1e-8 or y + height / 2 > 1 + 1e-8):
            raise ValueError(f"Caja YOLO fuera de límites: {path}")
        boxes.append([cls, x, y, width, height])
    return boxes
