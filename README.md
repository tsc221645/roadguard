# RoadGuard AI

RoadGuard AI es una plataforma de visión por computadora para detectar daños visibles en el pavimento a partir de imágenes y videos. El sistema combina un detector de objetos, evidencia visual y GPS aproximado para ayudar a priorizar inspecciones viales.

## Propósito

Las inspecciones manuales de calles consumen tiempo y pueden producir registros difíciles de comparar. RoadGuard propone un flujo reproducible que convierte un recorrido en video en incidencias revisables, sin presentar las predicciones como certificaciones estructurales ni reemplazar la validación humana.

El proyecto compara cinco iteraciones de YOLO11 (A–E) sobre el mismo test. El modelo B es el predeterminado en la app. Se reportan mAP, precisión, recall, falsas alarmas, omisiones y un costo ilustrativo bajo supuestos explícitos; el tiempo de revisión humana no se midió.

## Objetivos

- Detectar cuatro tipos de daño: D00, D10, D20 y D40.
- Mantener conjuntos de entrenamiento, validación y prueba trazables, sin fugas entre grupos.
- Procesar video con GPX cuando exista; si no hay GPS válido, dejar la ubicación como no disponible.
- Asociar cada detección con confianza, evidencia visual, índice de fotograma y ubicación aproximada cuando exista.
- Comparar cinco modelos bajo métricas técnicas y una función de valor económico.
- Proporcionar una base reproducible para una interfaz local para revisar evidencia y descargar resultados.

## Casos de uso

### Inspección asistida

Un operador carga un recorrido en video y examina las detecciones propuestas y descarga el reporte. La interfaz aún no registra confirmaciones o rechazos individuales.

### Priorización de mantenimiento

Un supervisor puede consultar el resumen por clase y la ubicación aproximada de la cámara cuando está disponible. La app no agrupa detecciones repetidas en incidencias únicas.

### Evaluación de modelos

El equipo compara A–E con el mismo test, revisa errores y observa cómo cambian el costo esperado y la carga de revisión.

### Demostración sin GPS

El sistema puede procesar video sin GPX. Las detecciones siguen disponibles con evidencia visual; la ubicación queda como no disponible salvo que el video incluya GPS utilizable.

## Flujo del sistema

```text
Imagen o video + GPX opcional para video
        ↓
Validación y extracción de frames
        ↓
Detector YOLO ajustado
        ↓
Detecciones por imagen o fotograma; ubicación si existe
        ↓
Evidencia anotada, resumen por clase y JSON descargable
```

## Datos y modelo

El entrenamiento utiliza RDD2022, con anotaciones Pascal VOC convertidas a YOLO. El dataset procesado contiene 23,764 muestras elegibles después de excluir archivos sin anotación válida, clases fuera del alcance y dos copias exactas con anotaciones contradictorias.

Los splits congelados están separados por paquete de origen:

- Train: 20,773 imágenes.
- Validation: 1,919 imágenes de China Drone.
- Test: 1,072 imágenes de Czech.

La separación por país/dominio favorece una evaluación conservadora de transferencia, aunque no representa todavía una prueba local en Guatemala.

## Estado actual

- Dataset descargado, extraído e inventariado.
- Conversión VOC → YOLO completada.
- EDA y revisión de integridad completados.
- Splits congelados y validados mediante hashes.
- Smoke test ejecutado en la RTX 3050.
- Entrenamiento A, B, D y E completado; C se detuvo tras 22 épocas.
- Evaluación A–E sobre el test congelado completada.
- Selección de umbral por F1, API e interfaz local: implementadas. El umbral económico 0.05 es exploratorio y no sustituye la configuración predeterminada 0.10.

## Requisitos mínimos

### Inferencia: imágenes y videos

- Windows 10/11 de 64 bits.
- CPU de cuatro núcleos.
- 8 GB de RAM.
- 5 GB de espacio libre, además del tamaño de los videos que se procesen.
- Python 3.11–3.13 y el entorno Conda `dm1`.
- Navegador moderno para la interfaz local.
- El checkpoint `artifacts/runs/model_b/weights/best.pt`.

La GPU no es obligatoria para inferencia: sin CUDA, configura `$env:ROADGUARD_DEVICE = "cpu"` antes de iniciar la app. Una GPU NVIDIA con al menos 4 GB de VRAM reduce considerablemente el tiempo de procesamiento. La RTX 3050 de 8 GB usada en desarrollo cumple los requisitos recomendados.

### Entrenamiento

- GPU NVIDIA con 8 GB de VRAM recomendados.
- 16 GB de RAM.
- 30–50 GB de espacio libre en SSD.
- PyTorch con CUDA compatible y el entorno Conda `dm1`.

## Ejecución en Windows

Desde la raíz del repositorio en PowerShell, crea el entorno e instala PyTorch antes de `requirements.in`. El comando CUDA 12.6 corresponde a los binarios oficiales de PyTorch para Windows; requiere una GPU NVIDIA y un controlador compatible. `pillow-heif==0.22.0` ya está incluido en `requirements.in`.

```powershell
conda create -n dm1 python=3.13 -y
conda activate dm1
$env:PYTHONNOUSERSITE = "1"
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.in
python -c "import torch, ultralytics, fastapi, pillow_heif; print('CUDA:', torch.cuda.is_available())"
python scripts/check_environment.py --output artifacts/reports/environment_local.json
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Abre `http://127.0.0.1:8000`. Sin GPU NVIDIA, usa `python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu` en lugar de la línea CUDA y configura `$env:ROADGUARD_DEVICE = "cpu"` antes de iniciar `uvicorn`. El checkpoint `artifacts/runs/model_b/weights/best.pt` debe estar disponible; la carga de imágenes originales del dataset no es necesaria para usar la app. [PyTorch documenta ambos índices de instalación](https://pytorch.org/get-started/previous-versions/).

Entrenamiento reproducible:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n dm1 python -m ml.train --name model_a --epochs 25
conda run -n dm1 python -m ml.train --name model_b --epochs 25 --lr0 0.0005 --degrees 3
```

Preparación del dataset:

```powershell
python scripts/download_rdd.py --workers 4 --extract
python -m ml.inspect_rdd --root data/raw/RDD2022
python -m ml.convert_voc_to_yolo --root data/raw/RDD2022
```

`configs/classes.yaml` es la única correspondencia de clases. Las imágenes del dataset, los videos y los archivos de `artifacts/app/` están excluidos de Git; algunos reportes y checkpoints están versionados.

## Uso del modelo entrenado

Para reproducir las cuentas operativas del test se necesitan las imágenes y etiquetas RDD2022 preparadas localmente. Desde `dm1`, ejecuta `python -m ml.operational_metrics --model artifacts/runs/model_b/weights/best.pt --name model_b_test_repro --split test --conf 0.10 --iou 0.50 --imgsz 640`. `python -m ml.value_analysis --model-b artifacts/evaluations/model_b_test_repro/operational.json --output artifacts/evaluations/value_repro.json` recalcula el escenario Q100/Q5 con el reporte A existente. El barrido de validación A/B se conserva en `artifacts/evaluations/validation_threshold_cost.json`; sus cuentas se regeneran con `--split val`, cada modelo y cada umbral de la tabla del informe, usando nombres de salida distintos. `ml.operational_metrics` admite `--device cpu` si no hay CUDA; las cifras de tiempo de CPU pueden diferir mucho de las mediciones en RTX 3050.

La calibración en el conjunto de validación seleccionó `conf=0.10` para ambos modelos. El modelo B se establece como opción predeterminada porque obtuvo un recall y un F1 ligeramente superiores.

Para ejecutar inferencia sobre una imagen, carpeta o video desde el entorno Conda `dm1`:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n dm1 python -m ml.predict --source ruta\al\recorrido.mp4 --device 0 --report
```

Por defecto usa `artifacts/runs/model_b/weights/best.pt` y guarda los resultados en `artifacts/predictions/latest`. `--report` genera `detections.json` con clase, confianza y coordenadas `xyxy` de las cajas **dentro de la imagen**, no GPS. La geolocalización EXIF, metadatos de video y GPX corresponde al flujo de la aplicación web. Se puede cambiar el modelo con `--model`, el nombre de salida con `--name` y exportar etiquetas con `--save-txt --save-conf`.

### Interfaz local

La interfaz web local permite cargar imágenes o videos, seleccionar el umbral de confianza, visualizar la evidencia y descargar el reporte JSON. Con `dm1` activado, se ejecuta así en GPU; para CPU, sustituye `"0"` por `"cpu"`:

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:ROADGUARD_DEVICE = "0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Luego abre `http://127.0.0.1:8000`. Los archivos generados se guardan en `artifacts/app`. El backend limita las cargas a 500 MB y no envía archivos fuera del equipo.

### Cómo probar una imagen o video

1. Inicia la interfaz con el comando anterior.
2. Abre `http://127.0.0.1:8000`.
3. Arrastra una imagen o video al área de carga.
4. Usa `0.10` como confianza inicial; es el umbral seleccionado en validación.
5. Presiona **Analizar archivo**.
6. Revisa la evidencia visual y el resumen por clase.
7. Descarga `detections.json` si necesitas conservar las detecciones.

La interfaz muestra el avance de la carga y, durante la inferencia, los fotogramas procesados y las detecciones encontradas. Se puede recargar la pestaña y recuperar el progreso mientras el servidor siga activo. Los trabajos se procesan de uno en uno para compartir la GPU. Los resultados visuales se guardan en `artifacts/app/results/<job_id>/prediction` y el reporte en `artifacts/app/results/<job_id>/detections.json`. En videos, el conteo corresponde a detecciones por fotograma; un mismo daño puede aparecer varias veces. Si el navegador no reproduce el video generado, descárgalo y ábrelo con un reproductor local.

### Ubicación aproximada

En imágenes JPG, PNG, WebP, HEIC o HEIF, la app lee coordenadas GPS EXIF si están presentes. HEIC/HEIF requiere `pillow-heif==0.22.0` en el entorno `dm1`; el original se conserva y solo se convierte una copia temporal a PNG para el detector. Al pulsar **Mostrar mapa con pin**, carga un mapa integrado centrado en la ubicación aproximada de la cámara. En videos, lee la ubicación y fecha incorporadas cuando `ffprobe` está disponible. Puedes adjuntar un GPX con puntos fechados y, si el video perdió su fecha de creación, indicar la hora local de inicio de grabación. El GPX se sincroniza por fotograma usando la velocidad del video; solo se asignan coordenadas cuando dos puntos contiguos están separados por 30 segundos o menos. El JSON incluye `location` por fotograma, `location_status` y `located_frames`. La interfaz muestra una coordenada de referencia y un enlace a OpenStreetMap. Al pulsar el botón del mapa o abrir el enlace, el navegador comparte esa coordenada con OpenStreetMap; se requiere conexión a Internet. Las coordenadas indican la posición aproximada de la cámara, no la posición exacta del daño ni toda la calle. Si no hay GPS o no se puede sincronizar el GPX, aparece «Ubicación no disponible». Los archivos cargados y GPX quedan en `artifacts/app/uploads`, que está excluido de Git.

Las clases mostradas por la interfaz son:

- `D00`: grieta longitudinal.
- `D10`: grieta transversal.
- `D20`: grieta tipo piel de cocodrilo.
- `D40`: bache.

Las predicciones son una ayuda para revisión y deben confirmarse visualmente; no representan una certificación estructural.

## Alcance y limitaciones

RoadGuard es un MVP académico local. No estima profundidad ni costo real de reparación, no ofrece localización centimétrica, no reemplaza una inspección profesional y no envía reportes automáticamente a instituciones públicas. El GPS representa aproximadamente la posición de la cámara, no necesariamente la ubicación exacta del defecto.

El código se publica bajo AGPL-3.0. Las licencias y atribuciones del dataset, pesos y dependencias deben conservarse por separado.
