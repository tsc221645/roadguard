"""Backend local de RoadGuard para inferencia sobre imágenes y videos."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

PROJECT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT / "app" / "static"
ARTIFACTS_DIR = PROJECT / "artifacts" / "app"
UPLOADS_DIR = ARTIFACTS_DIR / "uploads"
RESULTS_DIR = ARTIFACTS_DIR / "results"
MODEL_PATH = PROJECT / "artifacts" / "runs" / "model_b" / "weights" / "best.pt"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv"}

app = FastAPI(title="RoadGuard API", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_model: YOLO | None = None
_queue: Queue[tuple[str, Path, Path, float, str, int, bool]] = Queue()
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()
_worker: Thread | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail=f"No se encontró el modelo: {MODEL_PATH}")
        _model = YOLO(str(MODEL_PATH))
    return _model


def update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        _jobs[job_id].update(changes)


def video_frame_count(path: Path) -> int | None:
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        return count if count > 0 else None
    finally:
        capture.release()


def process_job(job_id: str, input_path: Path, output_dir: Path, confidence: float,
                filename: str, size: int, is_video: bool) -> None:
    try:
        model = get_model()
        total_frames = video_frame_count(input_path) if is_video else 1
        update_job(job_id, status="processing", total_frames=total_frames)
        summary = {name: 0 for name in model.names.values()}
        items = []
        results = model.predict(source=str(input_path), conf=confidence, imgsz=640,
                                device=os.getenv("ROADGUARD_DEVICE", "0"),
                                project=str(output_dir), name="prediction", exist_ok=True,
                                save=True, stream=True, verbose=False)
        for frame_index, result in enumerate(results, start=1):
            rows = []
            if result.boxes is not None:
                boxes = result.boxes
                for coordinates, score, class_id in zip(
                    boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
                ):
                    class_id = int(class_id)
                    name = result.names[class_id]
                    summary[name] = summary.get(name, 0) + 1
                    rows.append({"class_id": class_id, "class_name": name,
                                 "confidence": round(float(score), 5),
                                 "xyxy": [round(float(value), 2) for value in coordinates]})
            items.append({"source": Path(result.path).name, "frame": frame_index,
                          "detections": rows})
            update_job(job_id, processed_frames=frame_index,
                       total_detections=sum(summary.values()))

        report = {"model": str(MODEL_PATH), "confidence_threshold": confidence,
                  "total_detections": sum(summary.values()),
                  "detections_by_class": summary, "items": items,
                  "job_id": job_id, "filename": filename, "bytes": size}
        rendered_dir = output_dir / "prediction"
        rendered = next((path for path in rendered_dir.iterdir() if path.is_file()), None)
        if rendered is None:
            raise RuntimeError("La inferencia no generó un resultado visual.")
        (output_dir / "detections.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        update_job(job_id, status="completed", progress=100,
                   total_frames=len(items) if total_frames is None else total_frames,
                   result={**report, "media_type": "video" if is_video else "image",
                           "result_url": f"/api/jobs/{job_id}/media",
                           "report_url": f"/api/jobs/{job_id}/report"})
    except Exception as error:
        update_job(job_id, status="failed", error=str(error))


def run_queue() -> None:
    while True:
        job = _queue.get()
        try:
            process_job(*job)
        finally:
            _queue.task_done()


def ensure_worker() -> None:
    global _worker
    with _jobs_lock:
        if _worker is None or not _worker.is_alive():
            _worker = Thread(target=run_queue, daemon=True, name="roadguard-inference")
            _worker.start()


@app.get("/", response_class=FileResponse)
def index() -> Path:
    return STATIC_DIR / "index.html"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_PATH.name, "model_available": MODEL_PATH.exists(), "device": os.getenv("ROADGUARD_DEVICE", "0")}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), confidence: float = Query(0.10, gt=0.0, lt=1.0)) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato no soportado. Usa JPG, PNG, MP4, MOV, AVI o MKV.")
    job_id = uuid.uuid4().hex[:12]
    input_dir, output_dir = UPLOADS_DIR / job_id, RESULTS_DIR / job_id
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"input{suffix}"
    size = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="El archivo supera el límite de 500 MB.")
                destination.write(chunk)
        get_model()
        is_video = suffix in {".mp4", ".mov", ".avi", ".mkv"}
        job = {"job_id": job_id, "status": "queued", "filename": file.filename,
               "processed_frames": 0, "total_frames": None, "total_detections": 0}
        with _jobs_lock:
            _jobs[job_id] = job
        ensure_worker()
        _queue.put((job_id, input_path, output_dir, confidence,
                    file.filename or input_path.name, size, is_video))
        return job
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Error durante la inferencia: {error}") from error
    finally:
        await file.close()


def job_path(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    path = RESULTS_DIR / job_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
    return path


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
    raise HTTPException(status_code=404, detail="Trabajo no encontrado o servidor reiniciado.")


@app.get("/api/jobs/{job_id}/report")
def report(job_id: str) -> FileResponse:
    path = job_path(job_id) / "detections.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Reporte no encontrado.")
    return FileResponse(path, media_type="application/json", filename="detections.json")


@app.get("/api/jobs/{job_id}/media")
def media(job_id: str) -> FileResponse:
    rendered_dir = job_path(job_id) / "prediction"
    rendered = next((path for path in rendered_dir.iterdir() if path.is_file()), None)
    if rendered is None:
        raise HTTPException(status_code=404, detail="Resultado visual no encontrado.")
    return FileResponse(rendered)
