from datetime import datetime, timezone

from app.location import gps_coordinates, gpx_location, read_gpx, parse_time


def test_exif_gps_conversion():
    tags = {"GPSLatitude": (14, 37, 30), "GPSLatitudeRef": "N",
            "GPSLongitude": (90, 30, 0), "GPSLongitudeRef": "W"}
    assert gps_coordinates(tags) == (14.625, -90.5)
    assert gps_coordinates({**tags, "GPSLatitudeRef": "X"}) is None


def test_gpx_interpolation_and_gap(tmp_path):
    path = tmp_path / "route.gpx"
    path.write_text('''<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>
    <trkpt lat="14.0" lon="-90.0"><time>2026-11-15T17:00:00Z</time></trkpt>
    <trkpt lat="14.1" lon="-90.1"><time>2026-11-15T17:00:10Z</time></trkpt>
    <trkpt lat="15.0" lon="-91.0"><time>2026-11-15T17:02:00Z</time></trkpt>
    </trkseg></trk></gpx>''', encoding="utf-8")
    points = read_gpx(path)
    assert len(points) == 3
    midpoint = gpx_location(points, parse_time("2026-11-15T17:00:05Z"))
    assert midpoint["latitude"] == 14.05
    assert midpoint["longitude"] == -90.05
    assert gpx_location(points, parse_time("2026-11-15T17:01:00Z")) is None
    assert gpx_location(points, datetime(2026, 11, 15, 16, tzinfo=timezone.utc)) is None
