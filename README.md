# RoadGuard AI

RoadGuard AI es una plataforma de visión por computadora para detectar daños visibles en el pavimento a partir de video. El sistema combina un detector de objetos, evidencia visual y GPS aproximado para ayudar a priorizar inspecciones viales.

## Propósito

Las inspecciones manuales de calles consumen tiempo y pueden producir registros difíciles de comparar. RoadGuard propone un flujo reproducible que convierte un recorrido en video en incidencias revisables, sin presentar las predicciones como certificaciones estructurales ni reemplazar la validación humana.

El proyecto compara dos versiones ajustadas de un detector YOLO utilizando el mismo conjunto de prueba. Además de mAP, precision y recall, se analizarán falsas alarmas, daños omitidos, tiempo de revisión y costo esperado bajo supuestos explícitos.

## Objetivos

- Detectar cuatro tipos de daño: D00, D10, D20 y D40.
- Mantener conjuntos de entrenamiento, validación y prueba trazables, sin fugas entre grupos.
- Procesar video con GPX cuando exista; si no hay GPS válido, conservar la incidencia sin inventar coordenadas.
- Asociar cada detección con confianza, evidencia visual, instante del video y ubicación aproximada.
- Comparar dos modelos bajo métricas técnicas y una función de valor económico.
- Proporcionar una base reproducible para una futura aplicación local de revisión y priorización.

## Casos de uso

### Inspección asistida

Un operador carga un recorrido en video y revisa las detecciones propuestas por el modelo, confirmándolas o rechazándolas.

### Priorización de mantenimiento

Un supervisor consulta incidencias agrupadas por tipo de daño y ubicación aproximada para ordenar revisiones de campo.

### Evaluación de modelos

El equipo compara modelos A/B con el mismo test, revisa errores y observa cómo cambian el costo esperado y la carga de revisión.

### Demostración sin GPS

El sistema puede procesar video sin GPX. Las detecciones siguen disponibles con evidencia visual, pero su posición queda explícitamente como no disponible.

## Flujo del sistema

```text
Video + GPX opcional
        ↓
Validación y extracción de frames
        ↓
Detector YOLO ajustado
        ↓
Consolidación temporal de detecciones
        ↓
Incidencias con evidencia y GPS aproximado
        ↓
Revisión humana, métricas y exportación
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
- Entrenamiento completo de los modelos A y B completado.
- Evaluación A/B sobre el test congelado completada.
- Selección de umbral, API y frontend: pendientes.

## Requisitos mínimos

### Inferencia: imágenes y videos

- Windows 10/11 de 64 bits.
- CPU de cuatro núcleos.
- 8 GB de RAM.
- 5 GB de espacio libre, además del tamaño de los videos que se procesen.
- Python 3.11–3.13 y el entorno Conda `dm1`.
- Navegador moderno para la interfaz local.
- El checkpoint `artifacts/runs/model_b/weights/best.pt`.

La GPU no es obligatoria para inferencia, pero una GPU NVIDIA con al menos 4 GB de VRAM reduce considerablemente el tiempo de procesamiento. La RTX 3050 de 8 GB usada en desarrollo cumple los requisitos recomendados.

### Entrenamiento

- GPU NVIDIA con 8 GB de VRAM recomendados.
- 16 GB de RAM.
- 30–50 GB de espacio libre en SSD.
- PyTorch con CUDA compatible y el entorno Conda `dm1`.

## Ejecución en Windows

Para el pipeline de datos se requiere Python 3.13 con las dependencias correspondientes. Para entrenamiento en la RTX 3050 se utiliza el entorno Conda `dm1`, que ya tiene PyTorch con CUDA y Ultralytics.

```bash
source .venv/bin/activate
python scripts/check_environment.py --output artifacts/reports/environment_local.json
```

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

`configs/classes.yaml` es la única correspondencia de clases. Los datos, pesos, videos y artefactos grandes están excluidos de Git.

## Uso del modelo entrenado

La calibración en el conjunto de validación seleccionó `conf=0.10` para ambos modelos. El modelo B se establece como opción predeterminada porque obtuvo un recall y un F1 ligeramente superiores.

Para ejecutar inferencia sobre una imagen, carpeta o video desde el entorno Conda `dm1`:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n dm1 python -m ml.predict --source ruta\al\recorrido.mp4 --device 0 --report
```

Por defecto usa `artifacts/runs/model_b/weights/best.pt` y guarda los resultados en `artifacts/predictions/latest`. `--report` genera `detections.json` con clase, confianza y coordenadas. Se puede cambiar el modelo con `--model`, el nombre de salida con `--name` y exportar etiquetas con `--save-txt --save-conf`.

### Interfaz local

La interfaz web local permite cargar imágenes o videos, seleccionar el umbral de confianza, visualizar la evidencia y descargar el reporte JSON. Se ejecuta con el entorno `dm1`:

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

Las clases mostradas por la interfaz son:

- `D00`: grieta longitudinal.
- `D10`: grieta transversal.
- `D20`: grieta tipo piel de cocodrilo.
- `D40`: bache.

Las predicciones son una ayuda para revisión y deben confirmarse visualmente; no representan una certificación estructural.

## Alcance y limitaciones

RoadGuard es un MVP académico local. No estima profundidad ni costo real de reparación, no ofrece localización centimétrica, no reemplaza una inspección profesional y no envía reportes automáticamente a instituciones públicas. El GPS representa aproximadamente la posición de la cámara, no necesariamente la ubicación exacta del defecto.

El código se publica bajo AGPL-3.0. Las licencias y atribuciones del dataset, pesos y dependencias deben conservarse por separado.
