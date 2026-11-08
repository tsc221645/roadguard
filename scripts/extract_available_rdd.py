"""Extrae países completos de los bloques ya descargados, sin tocar la descarga.

Valida CRC32 del miembro exterior y CRC de cada archivo interior. El MD5 del
paquete completo sigue pendiente hasta finalizar download_rdd.py.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import tempfile
import zipfile
import zlib

from scripts.download_rdd import extract

BLOCK_SIZE = 32 * 1024 * 1024


def read_at(directory, start, size):
    data = bytearray()
    while size:
        index, offset = divmod(start, BLOCK_SIZE)
        with (directory / f"{index:05d}.part").open("rb") as stream:
            stream.seek(offset)
            block = stream.read(min(size, BLOCK_SIZE - offset))
        if not block:
            raise ValueError("Bloque completo truncado")
        data.extend(block)
        start += len(block)
        size -= len(block)
    return bytes(data)


def extract_available(parts, output, countries):
    parts, output = Path(parts), Path(output)
    available = 0
    index = 0
    while (part := parts / f"{index:05d}.part").exists():
        size = part.stat().st_size
        available += size
        index += 1
        if size < BLOCK_SIZE:
            break
    position, extracted = 0, []
    while position + 30 <= available:
        header = struct.unpack("<4s5H3I2H", read_at(parts, position, 30))
        signature, _, flags, method, _, _, crc, compressed, _, name_size, extra_size = header
        if signature != b"PK\x03\x04":
            break
        if flags & 8 or method != 0 or compressed == 0xffffffff:
            # No adivinar tamaños con data descriptors, compresión exterior o ZIP64.
            break
        if position + 30 + name_size + extra_size > available:
            break
        name = read_at(parts, position + 30, name_size).decode("utf-8")
        filename = PurePosixPath(name)
        if filename.is_absolute() or ".." in filename.parts or "\\" in name:
            raise ValueError("Ruta exterior no permitida")
        start = position + 30 + name_size + extra_size
        end = start + compressed
        if end > available:
            break
        country = filename.stem
        destination = output / country
        if filename.suffix.lower() == ".zip" and country in countries and not destination.exists():
            output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".member-", dir=output) as temporary:
                archive = Path(temporary) / filename.name
                checksum, digest = 0, hashlib.sha256()
                with archive.open("wb") as stream:
                    offset = start
                    while offset < end:
                        block = read_at(parts, offset, min(8 * 1024 * 1024, end - offset))
                        checksum = zlib.crc32(block, checksum)
                        digest.update(block)
                        stream.write(block)
                        offset += len(block)
                if checksum & 0xffffffff != crc:
                    raise ValueError(f"CRC exterior incorrecto: {name}")
                with zipfile.ZipFile(archive) as source:
                    if source.testzip() is not None:
                        raise ValueError(f"CRC interior incorrecto: {name}")
                extract(archive, destination, expand_nested=False)
                record = {"outer_member": name, "archive_sha256": digest.hexdigest(),
                          "outer_crc32": f"{crc:08x}", "inner_crc_verified": True,
                          "full_archive_md5": "pending"}
                (destination / "source_integrity.json").write_text(json.dumps(record, indent=2) + "\n")
            extracted.append(country)
        position = end
    return {"contiguous_bytes_available": available, "extracted": extracted,
            "full_download_complete": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=Path, default=Path("data/raw/downloads/.rdd_parts"))
    parser.add_argument("--output", type=Path, default=Path("data/raw/available"))
    parser.add_argument("--countries", nargs="+", default=["China_MotorBike", "Czech", "India", "Japan", "United_States"])
    args = parser.parse_args()
    print(json.dumps(extract_available(args.parts, args.output, set(args.countries)), indent=2))


if __name__ == "__main__":
    main()
