"""Pruebas sintéticas: validan el pipeline, no son resultados de RDD2022."""
import io
import json
import hashlib
import zipfile

from PIL import Image
import pytest

from ml.build_splits import assign_groups, build
from ml.convert_voc_to_yolo import convert
from ml.dataset import image_size, parse_voc, read_csv, read_labels, sha256, write_csv
from ml.eda import analyze
from ml.inspect_rdd import inspect
from ml.validate_dataset import validate
from scripts.download_rdd import extract
import scripts.download_rdd as downloader


def sample(root, stem, index=0, name="D00", coords=(10, 5, 90, 45), size=(100, 50)):
    images, annotations = root / "Japan/train/images", root / "Japan/train/annotations/xmls"
    images.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)
    image, xml = images / f"{stem}.jpg", annotations / f"{stem}.xml"
    Image.new("RGB", (100, 50), (index * 20 % 255, index * 37 % 255, 10)).save(image)
    box = "".join(f"<{key}>{value}</{key}>" for key, value in
                  zip(("xmin", "ymin", "xmax", "ymax"), coords))
    obj = f"<object><name>{name}</name><bndbox>{box}</bndbox></object>" if name else ""
    xml.write_text(f"<annotation><size><width>{size[0]}</width><height>{size[1]}</height></size>{obj}</annotation>")
    return image, xml


@pytest.mark.parametrize("name,index", [("D00", 0), ("D10", 1), ("D20", 2), ("D40", 3)])
def test_voc_coordinates_and_roundtrip(tmp_path, name, index):
    image, xml = sample(tmp_path, "sample", name=name)
    boxes, ignored = parse_voc(xml, image_size(image))
    assert ignored == []
    assert boxes[0] == pytest.approx([index, .5, .5, .8, .8])
    _, x, y, width, height = boxes[0]
    assert [(x-width/2)*100, (y-height/2)*50, (x+width/2)*100, (y+height/2)*50] == pytest.approx([10, 5, 90, 45])


@pytest.mark.parametrize("coords", [(-1, 5, 90, 45), (10, 5, 101, 45),
                                   (10, 5, 10, 45), (10, 5, 90, float("nan"))])
def test_bad_boxes_not_silently_clipped(tmp_path, coords):
    _, xml = sample(tmp_path, "bad", coords=coords)
    with pytest.raises(ValueError):
        parse_voc(xml, (100, 50))


def test_inventory_quarantine_and_negatives(tmp_path):
    raw = tmp_path / "raw"
    sample(raw, "good")
    sample(raw, "outside", coords=(0, 0, 101, 50))
    sample(raw, "unknown", name="D99")
    sample(raw, "empty", name=None)
    sample(raw, "mismatch", size=(99, 50))
    corrupt, _ = sample(raw, "corrupt")
    corrupt.write_bytes(b"broken jpg")
    image, xml = sample(raw, "orphan_image")
    xml.unlink()
    image, xml = sample(raw, "orphan_xml")
    image.unlink()
    _, broken_xml = sample(raw, "broken_xml")
    broken_xml.write_text("<annotation>")
    inventory = tmp_path / "inventory.csv"
    summary = inspect(raw, inventory)
    assert summary["status_counts"]["invalid"] == 4
    assert summary["status_counts"]["missing_xml"] == 1
    assert summary["status_counts"]["unpaired_xml"] == 1
    assert summary["status_counts"]["ignored_only"] == 1
    output = tmp_path / "converted"
    assert convert(raw, inventory, output)["accepted"] == 1
    assert convert(raw, inventory, tmp_path / "with_negatives", True)["accepted"] == 2
    assert len(list((output / "labels").glob("*.txt"))) == 1


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw"
    for i in range(24):
        sample(raw, f"Japan_{i:06d}", i, name=("D00", "D10", "D20", "D40")[i % 4])
    inventory, converted = tmp_path / "inventory.csv", tmp_path / "converted"
    inspect(raw, inventory)
    convert(raw, inventory, converted)
    groups = tmp_path / "groups.csv"
    rows = read_csv(converted / "converted.csv")
    write_csv(groups, [{"sample_id": row["sample_id"], "group_id": f"sequence_{i//4}"}
                       for i, row in enumerate(rows)], ["sample_id", "group_id"])
    return raw, inventory, converted, groups


def test_end_to_end_freeze_and_reproducibility(prepared, tmp_path):
    raw, inventory, converted, groups = prepared
    first, second = tmp_path / "first", tmp_path / "second"
    a = build(raw, converted, groups, first)
    b = build(raw, converted, groups, second)
    assert validate(first)["valid"]
    assert a == b
    for split in ("train", "val", "test"):
        assert (first / f"manifests/{split}.csv").read_bytes() == (second / f"manifests/{split}.csv").read_bytes()
    with pytest.raises(ValueError, match="existente"):
        build(raw, converted, groups, first)
    # Corrupción posterior: no debe pasar QA aunque conserve el nombre del archivo.
    label = next((first / "labels").rglob("*.txt"))
    label.write_text("0 .5 .5 .1 .1\n")
    with pytest.raises(ValueError, match="modificado"):
        validate(first)


def test_changed_source_rejected(prepared, tmp_path):
    raw, inventory, converted, groups = prepared
    next(raw.rglob("*.jpg")).write_bytes(b"modified")
    with pytest.raises(ValueError, match="modificada"):
        build(raw, converted, groups, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_frozen_manifest_edit_rejected(prepared, tmp_path):
    raw, _, converted, groups = prepared
    output = tmp_path / "dataset"
    build(raw, converted, groups, output)
    manifest = output / "manifests/train.csv"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifiesto congelado"):
        validate(output)


def test_cross_split_group_rejected_even_with_updated_manifest_hash(prepared, tmp_path):
    raw, _, converted, groups = prepared
    output = tmp_path / "dataset"
    build(raw, converted, groups, output)
    train = read_csv(output / "manifests/train.csv")
    val_path = output / "manifests/val.csv"
    val = read_csv(val_path)
    val[0]["group_id"] = train[0]["group_id"]
    write_csv(val_path, val, list(val[0]))
    report_path = output / "manifests/split_report.json"
    report = json.loads(report_path.read_text())
    report["splits"]["val"]["manifest_sha256"] = sha256(val_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cruzado"):
        validate(output)


def test_groups_required_and_minimum(prepared, tmp_path):
    raw, _, converted, groups = prepared
    rows = read_csv(groups)
    write_csv(groups, rows[:1], ["sample_id", "group_id"])
    with pytest.raises(ValueError, match="Faltan grupos"):
        build(raw, converted, groups, tmp_path / "dataset")
    all_rows = read_csv(converted / "converted.csv")
    write_csv(groups, [{"sample_id": row["sample_id"], "group_id": "one"} for row in all_rows],
              ["sample_id", "group_id"])
    with pytest.raises(ValueError, match="tres grupos"):
        build(raw, converted, groups, tmp_path / "dataset")


def test_duplicates_merge_groups_and_conflicting_labels_fail(tmp_path):
    rows = [{"sample_id": "a", "sha256": "h1", "label_sha256": "l1"},
            {"sample_id": "b", "sha256": "h1", "label_sha256": "l1"},
            {"sample_id": "c", "sha256": "h2", "label_sha256": "l2"}]
    groups = tmp_path / "groups.csv"
    write_csv(groups, [{"sample_id": "a", "group_id": "route_a"},
                      {"sample_id": "b", "group_id": "route_b"},
                      {"sample_id": "c", "group_id": "route_b"}], ["sample_id", "group_id"])
    result, duplicates = assign_groups(rows, groups)
    assert duplicates == 1
    assert len({row["group_id"] for row in result}) == 1
    rows[1]["label_sha256"] = "different"
    with pytest.raises(ValueError, match="contradictorias"):
        assign_groups(rows, groups)


def test_reviewed_similarity_links_connect_different_group_ids(tmp_path):
    rows = [{"sample_id": "a", "sha256": "ha", "label_sha256": "la"},
            {"sample_id": "b", "sha256": "hb", "label_sha256": "lb"},
            {"sample_id": "c", "sha256": "hc", "label_sha256": "lc"}]
    groups = tmp_path / "groups.csv"
    write_csv(groups, [{"sample_id": r["sample_id"], "group_id": r["sample_id"]} for r in rows],
              ["sample_id", "group_id"])
    links = tmp_path / "similarity.csv"
    write_csv(links, [{"sample_a": "a", "sample_b": "c"}], ["sample_a", "sample_b"])
    result, duplicates = assign_groups(rows, groups, links)
    assert duplicates == 0
    mapped = {r["sample_id"]: r["group_id"] for r in result}
    assert mapped["a"] == mapped["c"]
    assert mapped["a"] != mapped["b"]


def test_eda_reads_converted_labels(prepared, tmp_path):
    raw, inventory, converted, _ = prepared
    report = analyze(raw, inventory, converted, tmp_path / "eda", per_class=2)
    assert report["class_instances"] == {"D00": 6, "D10": 6, "D20": 6, "D40": 6}
    assert all(len(samples) == 2 for samples in report["overlays"].values())
    assert len(list((tmp_path / "eda").glob("overlays_*.jpg"))) == 4


def test_xml_dtd_rejected(tmp_path):
    _, xml = sample(tmp_path, "sample")
    xml.write_text('<!DOCTYPE annotation [<!ENTITY x SYSTEM "file:///etc/passwd">]>' + xml.read_text())
    with pytest.raises(ValueError, match="DTD"):
        parse_voc(xml, (100, 50))


def test_zip_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Ruta ZIP"):
        extract(archive, tmp_path / "unpacked")
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / "unpacked").exists()


def test_official_nested_zip_layout(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as stream:
        stream.writestr("Japan/train/images/sample.jpg", b"fixture")
    archive = tmp_path / "outer.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("RDD2022/Japan.zip", inner.getvalue())
    extract(archive, tmp_path / "unpacked")
    assert (tmp_path / "unpacked/RDD2022/Japan/Japan/train/images/sample.jpg").read_bytes() == b"fixture"


def test_download_resumes_partial_block_and_checks_md5(tmp_path, monkeypatch):
    payload = b"a complete fake archive"
    record = {"id": downloader.FILE_ID, "size": len(payload), "download_url": "https://example.invalid/file",
              "computed_md5": hashlib.md5(payload).hexdigest()}
    monkeypatch.setattr(downloader, "fetch_json", lambda _: {"files": [record], "license": {"name": "CC BY 4.0"}})
    partial = tmp_path / ".rdd_parts"
    partial.mkdir()
    (partial / "00000.tmp").write_bytes(payload[:5])
    requests = []
    def response(request, timeout):
        requests.append(request.get_header("Range"))
        result = io.BytesIO(payload[5:])
        result.status = 206
        result.headers = {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}"}
        return result
    monkeypatch.setattr(downloader, "urlopen", response)
    archive = downloader.download(tmp_path, workers=1)
    assert archive.read_bytes() == payload
    assert requests == [f"bytes=5-{len(payload)-1}"]
    assert not partial.exists()
    # Una segunda ejecución no vuelve a bajar un archivo ya verificado.
    assert downloader.download(tmp_path, workers=1) == archive
    assert len(requests) == 1


@pytest.mark.parametrize("line", ["0 .1 .5 .8 .1", "4 .5 .5 .1 .1", "0 nan .5 .1 .1"])
def test_bad_yolo_labels_rejected(tmp_path, line):
    path = tmp_path / "label.txt"
    path.write_text(line)
    with pytest.raises(ValueError):
        read_labels(path)
