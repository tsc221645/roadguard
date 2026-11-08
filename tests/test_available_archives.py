"""Lectura de países completos sin confundirlos con descarga general completa."""
import io
import json
import zipfile

import pytest

import scripts.extract_available_rdd as available


def nested_archive():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as archive:
        archive.writestr("Czech/train/annotations/xmls/a.xml", "<annotation/>")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("RDD2022/Czech.zip", inner.getvalue())
        archive.writestr("RDD2022/India.zip", inner.getvalue())
    return outer.getvalue()


def write_parts(directory, payload, block_size):
    directory.mkdir()
    for index, offset in enumerate(range(0, len(payload), block_size)):
        (directory / f"{index:05d}.part").write_bytes(payload[offset:offset + block_size])


def test_extract_complete_member_from_partial_outer_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(available, "BLOCK_SIZE", 64)
    payload = nested_archive()
    # Primer miembro completo; segundo incompleto. No requiere directorio central.
    parts = tmp_path / "parts"
    write_parts(parts, payload[:len(payload) // 2], 64)
    output = tmp_path / "available"
    result = available.extract_available(parts, output, {"Czech", "India"})
    assert result["extracted"] == ["Czech"]
    assert not result["full_download_complete"]
    assert (output / "Czech/Czech/train/annotations/xmls/a.xml").exists()
    record = json.loads((output / "Czech/source_integrity.json").read_text())
    assert record["inner_crc_verified"]
    assert record["full_archive_md5"] == "pending"
    assert available.extract_available(parts, output, {"Czech"})["extracted"] == []


def test_corrupt_member_never_published(tmp_path, monkeypatch):
    monkeypatch.setattr(available, "BLOCK_SIZE", 64)
    payload = bytearray(nested_archive())
    payload[100] ^= 1
    parts = tmp_path / "parts"
    write_parts(parts, payload, 64)
    output = tmp_path / "available"
    with pytest.raises(ValueError, match="CRC exterior"):
        available.extract_available(parts, output, {"Czech"})
    assert not (output / "Czech").exists()


def test_out_of_order_blocks_not_read_as_contiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(available, "BLOCK_SIZE", 64)
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "00001.part").write_bytes(b"x" * 64)
    result = available.extract_available(parts, tmp_path / "output", {"Czech"})
    assert result["contiguous_bytes_available"] == 0
    assert result["extracted"] == []
