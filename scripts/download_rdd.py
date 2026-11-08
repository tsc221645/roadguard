"""Descarga oficial Figshare con reanudación por bloques, MD5 y extracción segura.

Uso: python scripts/download_rdd.py [--workers 4] [--extract]
Solo usa biblioteca estándar. Los ZIP/datos quedan dentro de data/ (ignorado).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import time
from urllib.request import Request, urlopen
import zipfile

ARTICLE = "https://api.figshare.com/v2/articles/21431547"
FILE_ID = 38030910


def fetch_json(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def md5sum(path):
    digest = hashlib.md5()  # Hash de integridad publicado por Figshare, no uso criptográfico.
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(archive, destination, expand_nested=True):
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"No se sobrescribe destino existente: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                path = (stage / item.filename).resolve()
                if not path.is_relative_to(stage.resolve()) or "\\" in item.filename:
                    raise ValueError(f"Ruta ZIP no permitida: {item.filename}")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Enlaces simbólicos no permitidos en ZIP")
                if item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as src, path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
        # Figshare distribuye un ZIP exterior que contiene los ZIP de cada país.
        if expand_nested:
            for nested in sorted(stage.rglob("*.zip")):
                extract(nested, nested.with_suffix(""), expand_nested=False)
        stage.rename(destination)
    except BaseException:
        shutil.rmtree(stage)
        raise


def download(directory, workers=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = fetch_json(ARTICLE)
    record = next(f for f in metadata["files"] if f["id"] == FILE_ID)
    (directory / "figshare_metadata.json").write_text(json.dumps(metadata, indent=2))
    target = directory / "RDD2022.zip"
    expected = record["computed_md5"]
    if target.exists() and target.stat().st_size == record["size"] and md5sum(target) == expected:
        print("Archivo existente verificado", flush=True)
        return target
    parts = directory / ".rdd_parts"
    parts.mkdir(exist_ok=True)
    block_size = 32 * 1024 * 1024
    blocks = [(index, start, min(start + block_size, record["size"]) - 1)
              for index, start in enumerate(range(0, record["size"], block_size))]
    # Reutilizar prefijo de una descarga secuencial interrumpida. MD5 final obligatorio.
    if target.exists() and target.stat().st_size < record["size"]:
        with target.open("rb") as existing:
            for index, start, end in blocks:
                if end >= target.stat().st_size:
                    break
                part = parts / f"{index:05d}.part"
                if not part.exists():
                    existing.seek(start)
                    part.write_bytes(existing.read(end - start + 1))
    def get_block(block):
        index, start, end = block
        part = parts / f"{index:05d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return index
        for attempt in range(4):
            try:
                temporary = part.with_suffix(".tmp")
                received = temporary.stat().st_size if temporary.exists() else 0
                if received > end - start + 1:
                    temporary.unlink()
                    received = 0
                if received == end - start + 1:
                    temporary.replace(part)
                    return index
                offset = start + received
                request = Request(record["download_url"], headers={"Range": f"bytes={offset}-{end}"})
                with urlopen(request, timeout=60) as response:
                    expected_range = f"bytes {offset}-{end}/{record['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("Servidor no respeta Range; se rechaza una descarga incompleta")
                    with temporary.open("ab") as output:
                        shutil.copyfileobj(response, output)
                if temporary.stat().st_size != end - start + 1:
                    raise ValueError("Tamaño de bloque incorrecto")
                temporary.replace(part)
                return index
            except (OSError, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    print(f"Descarga oficial: {record['size']} bytes, licencia {metadata['license']['name']}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(get_block, block) for block in blocks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 10 == 0 or count == len(blocks):
                print(f"Bloques completos: {count}/{len(blocks)}", flush=True)
    assembled = directory / "RDD2022.assembling"
    digest = hashlib.md5()
    with assembled.open("wb") as output:
        for index, _, _ in blocks:
            with (parts / f"{index:05d}.part").open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
    if digest.hexdigest() != expected:
        assembled.unlink()
        raise ValueError("MD5 no coincide con Figshare; no extraer. Revisar/eliminar bloques antes de reintentar")
    assembled.replace(target)
    shutil.rmtree(parts)
    print(f"Archivo verificado MD5={expected}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/raw/downloads"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--destination", type=Path, default=Path("data/raw/RDD2022"))
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers debe estar entre 1 y 16")
    archive = download(args.directory, args.workers)
    if args.extract:
        extract(archive, args.destination)
        print(f"Datos extraídos en {args.destination}", flush=True)


if __name__ == "__main__":
    main()
