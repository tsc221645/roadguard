FROM python:3.13-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ROADGUARD_DEVICE=cpu \
    YOLO_CONFIG_DIR=/home/roadguard/.config/Ultralytics

WORKDIR /app

# ffprobe reads video timestamps and GPS metadata; OpenCV needs the GL libraries.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.app.lock.txt ./
RUN python -m pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements.app.lock.txt

RUN useradd --create-home --uid 10001 roadguard \
    && mkdir -p /app/artifacts/app /home/roadguard/.config/Ultralytics \
    && chown -R roadguard:roadguard /app/artifacts/app /home/roadguard

COPY --chown=roadguard:roadguard app/ ./app/
COPY --chown=roadguard:roadguard artifacts/runs/model_b/weights/best.pt ./artifacts/runs/model_b/weights/best.pt

USER roadguard
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
