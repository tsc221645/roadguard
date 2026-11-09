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

La calibraciÃ³n sobre validaciÃ³n seleccionÃ³ `conf=0.10` para ambos modelos; el modelo B se toma como opciÃ³n predeterminada por su mejor recall y F1 ligeramente superior.

Para ejecutar inferencia sobre una imagen, carpeta o video desde el entorno Conda `dm1`:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n dm1 python -m ml.predict --source ruta\al\recorrido.mp4 --device 0 --report
```

Por defecto usa `artifacts/runs/model_b/weights/best.pt` y guarda los resultados en `artifacts/predictions/latest`. `--report` genera `detections.json` con clase, confianza y coordenadas. Se puede cambiar el modelo con `--model`, el nombre de salida con `--name` y exportar etiquetas con `--save-txt --save-conf`.

## Alcance y limitaciones

RoadGuard es un MVP académico local. No estima profundidad ni costo real de reparación, no ofrece localización centimétrica, no reemplaza una inspección profesional y no envía reportes automáticamente a instituciones públicas. El GPS representa aproximadamente la posición de la cámara, no necesariamente la ubicación exacta del defecto.

El código se publica bajo AGPL-3.0. Las licencias y atribuciones del dataset, pesos y dependencias deben conservarse por separado.
