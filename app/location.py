"""Extract approximate camera coordinates from local media and GPX tracks."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ExifTags

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False


def heic_to_png(source: Path, destination: Path) -> None:
    """Decode for YOLO while keeping the uploaded HEIC intact for GPS extraction."""
    if not HEIC_AVAILABLE:
        raise RuntimeError("Falta pillow-heif en el entorno de la aplicación.")
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo else None
    except ValueError:
        return None


def gps_coordinates(tags: dict) -> tuple[float, float] | None:
    try:
        def decimal(values, ref):
            degrees, minutes, seconds = map(float, values)
            coordinate = degrees + minutes / 60 + seconds / 3600
            return -coordinate if ref in {"S", "W"} else coordinate

        lat_ref = tags["GPSLatitudeRef"].decode("ascii") if isinstance(tags["GPSLatitudeRef"], bytes) else str(tags["GPSLatitudeRef"])
        lon_ref = tags["GPSLongitudeRef"].decode("ascii") if isinstance(tags["GPSLongitudeRef"], bytes) else str(tags["GPSLongitudeRef"])
        if lat_ref.upper() not in {"N", "S"} or lon_ref.upper() not in {"E", "W"}:
            return None
        lat = decimal(tags["GPSLatitude"], lat_ref.upper())
        lon = decimal(tags["GPSLongitude"], lon_ref.upper())
        return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def location(lat: float, lon: float, source: str, timestamp: datetime | None = None) -> dict:
    value = {"latitude": round(lat, 7), "longitude": round(lon, 7), "source": source}
    if timestamp:
        value["timestamp"] = timestamp.isoformat()
    return value


def image_location(path: Path) -> dict | None:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            raw = exif.get_ifd(ExifTags.IFD.GPSInfo)
            tags = {ExifTags.GPSTAGS.get(key, key): value for key, value in raw.items()}
            coords = gps_coordinates(tags)
            if coords is None:
                return None
            return location(*coords, "EXIF GPS")
    except (OSError, KeyError, TypeError, ValueError):
        return None


ISO6709 = re.compile(r"^([+-]\d{2}(?:\.\d+)?)([+-]\d{3}(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?$")


def video_metadata(path: Path) -> tuple[dict | None, datetime | None]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, None
    try:
        result = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags:stream_tags", "-of", "json", str(path)],
                                capture_output=True, text=True, timeout=15, check=True)
        data = json.loads(result.stdout)
        tags = [data.get("format", {}).get("tags", {})] + [stream.get("tags", {}) for stream in data.get("streams", [])]
        start = None
        found = None
        for group in tags:
            normalized = {str(key).lower(): value for key, value in group.items()}
            start = start or parse_time(normalized.get("creation_time") or normalized.get("com.apple.quicktime.creationdate"))
            raw = normalized.get("com.apple.quicktime.location.iso6709") or normalized.get("location")
            match = ISO6709.fullmatch(str(raw or ""))
            if match:
                lat, lon = float(match[1]), float(match[2])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    found = location(lat, lon, "Metadatos del video")
                    break
        return found, start
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
        return None, None


def read_gpx(path: Path) -> list[tuple[datetime, float, float]]:
    points = []
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] not in {"trkpt", "rtept"}:
            continue
        try:
            lat, lon = float(element.attrib["lat"]), float(element.attrib["lon"])
            time_text = next((child.text for child in element if child.tag.rsplit("}", 1)[-1] == "time"), None)
            when = parse_time(time_text)
            if when and -90 <= lat <= 90 and -180 <= lon <= 180:
                points.append((when, lat, lon))
        except (KeyError, TypeError, ValueError):
            pass
        element.clear()
    return sorted(points)


def gpx_location(points: list[tuple[datetime, float, float]], when: datetime, max_gap_seconds: float = 30) -> dict | None:
    if not points:
        return None
    from bisect import bisect_left
    index = bisect_left(points, (when, -91, -181))
    if index < len(points) and points[index][0] == when:
        _, lat, lon = points[index]
        return location(lat, lon, "GPX", when)
    if 0 < index < len(points):
        before, after = points[index - 1], points[index]
        span = (after[0] - before[0]).total_seconds()
        if 0 < span <= max_gap_seconds:
            ratio = (when - before[0]).total_seconds() / span
            return location(before[1] + ratio * (after[1] - before[1]),
                            before[2] + ratio * (after[2] - before[2]), "GPX", when)
    return None
