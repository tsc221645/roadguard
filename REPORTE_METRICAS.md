# RoadGuard AI — reporte comparativo de modelos

Fecha: 13 de noviembre de 2026. Referencia: *Proyecto Final - Caso de uso real.pdf*, secciones 3 y 4.1–4.8. Este reporte usa los resultados locales de `artifacts/runs/` y `artifacts/evaluations/`.

## Resumen ejecutivo

Se entrenaron cinco iteraciones de detector de daños viales (A–E). El modelo **B, YOLO11n a 640 px**, sigue siendo la elección general para la aplicación: mAP50 **21.91 %**, mAP50–95 **7.54 %** y recall macro de Ultralytics **29.45 %** en el test. E obtiene mayor precisión macro (**33.93 %** frente a **32.36 %** de B) y un F1 derivado de las medias macro ligeramente mayor; sus mAP y recall son inferiores. El conjunto de test corresponde a Czech, fuera de los grupos de entrenamiento. Las cifras no demuestran desempeño en calles de Guatemala.

El análisis de valor por caja etiquetada, a confianza `0.10` e IoU `0.50`, da una ventaja ilustrativa de **Q370** de B sobre A en las **1,072 imágenes** de prueba, bajo supuestos de costo declarados abajo. El intervalo bootstrap pareado de 95 % es **Q−2,075 a Q2,795**: incluye cero. Por tanto, no hay evidencia suficiente para afirmar un ahorro económico real. No se extrapola a dinero por kilómetro ni por mes.

## Datos y protocolo

- RDD2022 procesado: 23,764 imágenes elegibles; train 20,773; validación 1,919 (China Drone); test 1,072 (Czech). Todas las evaluaciones de la tabla principal usan el mismo test a **640 px**. En test hay 1,745 cajas etiquetadas: D00 988, D10 399, D20 161 y D40 197.
- Clases: D00 grieta longitudinal, D10 transversal, D20 piel de cocodrilo y D40 bache/deterioro agrupado. La correspondencia está en `configs/classes.yaml`.
- Cada checkpoint `best.pt` se eligió durante entrenamiento según validación. A y B tienen selección documentada de confianza `0.10` en validación por F1. Para comparar los cinco, el conteo operacional usa **el mismo** `conf=0.10`, `IoU=0.50`, emparejamiento 1 a 1 por clase y resolución de 640 px. No se optimizó el umbral específico de C, D o E.
- `precision` y `recall` de la tabla de Ultralytics son métricas agregadas de su evaluador, calculadas con su procedimiento de evaluación. El F1 indicado allí se calcula a partir de esas dos medias; **no** debe confundirse con el F1 micro del conteo operacional a un umbral fijo.

## Enfoques e iteraciones

| Modelo | Enfoque | Épocas completas | Tiempo de entrenamiento | Mejor mAP50 / mAP50–95 en validación |
|---|---|---:|---:|---:|
| A | Línea base YOLO11n preentrenado, 640 px, optimizador `auto` | 25 | 2.06 h | 32.34 % / 16.10 % |
| B | YOLO11n preentrenado, 640 px, rotación de 3° | 25 | 2.08 h | 32.20 % / 14.85 % |
| C | YOLO11n, intención de 50 épocas; detenido tras 22; `auto` escogió MuSGD y una tasa efectiva alta | 22 | 2.19 h | 26.16 % / 12.42 % |
| D | Ajuste fino desde el mejor checkpoint de B; 768 px, AdamW `1e-4`, coseno y mosaic reducido | 15 | 1.54 h | 33.64 % / 16.12 % |
| E | YOLO11s preentrenado, 640 px, AdamW `1e-3`, coseno, imágenes en RAM | 30 | 2.92 h | 36.18 % / 17.92 % |

Los tiempos son los campos `time` de la última fila de cada `results.csv`, no horas facturadas. La validación favoreció D y E; el test no confirmó una mejora global. El parámetro `lr0=0.0005` solicitado para B y C quedó bajo `optimizer=auto`, que selecciona su propia tasa; por ello **no** se atribuye el resultado de B a una tasa `0.0005` aplicada. Las pérdidas finales de entrenamiento (`box / cls / dfl`) fueron: A `1.664 / 1.636 / 1.510`, B `1.657 / 1.637 / 1.520`, C `1.775 / 1.923 / 1.588`, D `1.640 / 1.639 / 1.569`, E `1.569 / 1.454 / 1.456`. No son comparables entre arquitecturas como una medida directa de valor; las curvas completas están en `artifacts/runs/<modelo>/results.csv`.

## Métricas técnicas en test

Métricas del evaluador Ultralytics en los 1,072 ejemplos a 640 px. mAP50 exige IoU 0.50; mAP50–95 promedia varios IoU.

| Modelo | Precisión | Recall | F1 derivado* | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| A | 34.38 % | 25.94 % | 29.57 % | 21.42 % | 7.39 % |
| **B** | 32.36 % | **29.45 %** | 30.84 % | **21.91 %** | **7.54 %** |
| C | 31.15 % | 24.61 % | 27.49 % | 20.29 % | 6.85 % |
| D | 31.69 % | 26.92 % | 29.11 % | 20.10 % | 7.08 % |
| E | **33.93 %** | 29.30 % | **31.45 %** | 21.64 % | 7.32 % |

\* `2PR/(P+R)` sobre precisión y recall agregados de Ultralytics; se incluye para comparación descriptiva, no equivale necesariamente al F1 por instancia de un umbral fijo. D también se evaluó a 768 px: precisión 34.03 %, recall 27.47 %, mAP50 21.72 % y mAP50–95 7.49 %. Aun así, B a 640 px conserva el mayor mAP global de las configuraciones evaluadas.

### mAP50–95 por clase en test

| Modelo | D00 | D10 | D20 | D40 |
|---|---:|---:|---:|---:|
| A | **11.83 %** | 9.35 % | 5.71 % | 2.66 % |
| B | 11.61 % | 8.89 % | **6.62 %** | 3.04 % |
| C | 11.60 % | 8.16 % | 5.10 % | 2.54 % |
| D (640 px) | 11.50 % | 8.47 % | 5.45 % | 2.90 % |
| E | 10.34 % | **9.88 %** | 5.95 % | **3.11 %** |

D40 es débil en todos los modelos. La mAP por clase no implica un número conocido de incidencias reparables; es una medida de calidad de cajas etiquetadas.

## Punto operativo y errores

Se ejecutó inferencia en el test con `conf=0.10`, `imgsz=640`, IoU de emparejamiento `0.50` y caja verdadera usada una sola vez. `TP`, `FP` y `FN` aquí son **cajas por imagen**, no daños distintos a lo largo de un recorrido. Los FP pueden incluir daños visibles que el dataset no etiquetó.

| Modelo | TP | FP | FN | Precisión micro | Recall micro | F1 micro | Alertas para revisar |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 529 | 807 | 1,216 | 39.60 % | 30.32 % | 34.34 % | 1,336 |
| **B** | **533** | 809 | **1,212** | **39.72 %** | **30.54 %** | **34.53 %** | 1,342 |
| C | 462 | **750** | 1,283 | 38.12 % | 26.48 % | 31.25 % | **1,212** |
| D | 490 | 826 | 1,255 | 37.23 % | 28.08 % | 32.02 % | 1,316 |
| E | 488 | 836 | 1,257 | 36.86 % | 27.97 % | 31.80 % | 1,324 |

Por clase, B recuperó `310/988` D00, `135/399` D10, `65/161` D20 y `23/197` D40; su recall D40 es **11.7 %**. A recuperó más D10 (`157/399`) y E más D20 (`78/161`) y D40 (`25/197`), pero E empeoró el total de daños omitidos. Las diferencias del conteo operacional frente a precisión/recall de Ultralytics obedecen a la definición del umbral y del agregado; no se deben mezclar en una misma fórmula de valor.

## Modelo de valor: escenario explícito, no ahorro observado

Proceso propuesto: una persona revisa cada alerta antes de enviar una inspección de campo. Para las 1,072 imágenes de test se plantea, **solo como supuesto ilustrativo**, Q100 por caja etiquetada omitida y Q5 por cada alerta revisada. Q5 equivaldrían a 2 minutos de revisión a Q150/hora, una tarifa hipotética. Q100 es un costo de omisión supuesto, pendiente de validar con una institución. No se cuenta dos veces la revisión de FP: el costo de Q5 aplica por igual a TP y FP. Se excluyen captura y cómputo monetario porque no se midieron potencia, tarifa, depreciación ni distancia recorrida.

`C_modelo = FN × Q100 + (TP + FP) × Q5 + C_captura + C_cómputo`

| Modelo | Costo de errores + revisión supuesto | Diferencia frente a A |
|---|---:|---:|
| A | Q128,280 | Referencia |
| **B** | **Q127,910** | **Q370 menos** |
| C | Q134,360 | Q6,080 más |
| D | Q132,080 | Q3,800 más |
| E | Q132,320 | Q4,040 más |

B reduce 4 FN y genera 6 alertas adicionales respecto de A; por eso `4 × Q100 − 6 × Q5 = Q370`. Si cada revisión tarda 2 minutos, B requeriría **12 minutos adicionales** de revisión en este lote, a cambio de 4 cajas etiquetadas menos omitidas. Es un vínculo técnico–operativo cuantificado, no un ahorro municipal demostrado. Si el costo relativo de omitir una caja cae por debajo de Q7.50, A sería más barato en este escenario (`4 × c_FN < 6 × Q5`).

Sensibilidad al costo supuesto de omisión, manteniendo Q5 por alerta: con Q50, Q80, Q100, Q120 y Q150, la ventaja nominal de B sobre A sería Q170, Q290, Q370, Q450 y Q570, respectivamente. Bootstrap pareado de 5,000 remuestras de imágenes, semilla 42: **IC percentil 95 % de la diferencia B frente a A: Q−2,075 a Q2,795**. El intervalo cruza cero y no captura incertidumbre en los costos supuestos ni en el cambio de dominio. No se extrapola a Q/km porque el test no identifica defectos únicos ni kilómetros recorridos.

El umbral `0.10` de A/B fue elegido por F1 en validación, **no por el costo supuesto**. Falta construir una curva costo vs. umbral en validación y congelar una selección por valor antes de cualquier evaluación confirmatoria adicional. El test ya se consultó varias veces durante el desarrollo A–E; esta limitación reduce su independencia como evidencia final.

## Costo computacional y experiencia de uso

Entrenamiento local: 2.06 h (A), 2.08 h (B), 2.19 h hasta detener C, 1.54 h (D) y 2.92 h (E), con RTX 3050 de 8 GB. Son tiempos de ejecución, **no costos monetarios**. E usó unos 24 GB de RAM para caché de imágenes y una arquitectura más grande; no mejoró el test global.

En la medición de inferencia con `ml.operational_metrics`, 1,072 imágenes a 640 px en RTX 3050, los tiempos reportados por Ultralytics para inferencia por imagen fueron:

| Modelo | Mediana | p95 | Tiempo de pared del lote | Throughput de pared |
|---|---:|---:|---:|---:|
| A | 7.40 ms | 17.03 ms | 17.02 s | 63.0 imágenes/s |
| B | 7.15 ms | 12.95 ms | 16.14 s | 66.4 imágenes/s |
| C | 7.00 ms | 13.63 ms | 15.92 s | 67.3 imágenes/s |
| D | 7.06 ms | 18.24 ms | 16.54 s | 64.8 imágenes/s |
| E | 9.13 ms | 13.05 ms | 17.91 s | 59.9 imágenes/s |

Las mediciones son exploratorias, no un benchmark controlado: el tiempo por imagen es solo inferencia reportada por Ultralytics; el tiempo de pared incluye carga, pre/postprocesamiento y calentamiento. No se registró consumo eléctrico ni costo monetario por predicción. La interfaz web local ya permite subir fotos/video, ver avance por fotogramas, descargar detecciones JSON y consultar el video procesado. Falta medir tiempo de tarea, tasa de éxito y correcciones con usuarios reales. Los contadores de video son por fotograma y no consolidan incidencias únicas.

## Conclusión y pendientes exigidos por la rúbrica

B es la opción operativa actual. Su ganancia técnica frente a A es pequeña y el escenario de valor tiene incertidumbre amplia; no se afirma retorno económico real. E ilustra rendimientos decrecientes: una arquitectura mayor y más tiempo de entrenamiento mejoraron validación, pero no el test global. Se eligió detección profunda porque el flujo necesita localizar múltiples daños de varias clases por imagen; una alternativa de reglas visuales o revisión manual aún no se evaluó bajo el mismo protocolo. La transferencia a Guatemala, la sensibilidad a iluminación y el costo de D40 omitidos siguen abiertos.

Para cerrar el trabajo final faltan: (1) anotar una muestra local independiente y medir errores por clase/condición; (2) medir kilómetros, tiempo de revisión manual y costo/hora reales; (3) construir la curva de valor por umbral en validación; (4) benchmark controlado de latencia y energía; (5) prueba de uso de la interfaz; (6) material de demo de 2–4 minutos y verificación de despliegue/reproducción. No hay resultados SHAP/Grad-CAM ni evidencia de usuarios reales; describir estos puntos como pendientes. La entrega escrita indicada por la rúbrica es el 18 de noviembre de 2026.

### Reproducción

En el entorno Conda `dm1`, desde la raíz del proyecto, con RDD2022 procesado y checkpoints locales:

```powershell
python -m ml.evaluate --model artifacts/runs/model_b/weights/best.pt --name model_b_test_repro --split test --imgsz 640
python -m ml.operational_metrics --model artifacts/runs/model_a/weights/best.pt --name model_a_test --conf 0.10 --iou 0.50
python -m ml.operational_metrics --model artifacts/runs/model_b/weights/best.pt --name model_b_test --conf 0.10 --iou 0.50
python -m ml.value_analysis
```

Para C, D y E, sustituir el nombre del modelo y del reporte en `ml.operational_metrics`. Los JSON fuente quedan en `artifacts/evaluations/<modelo>_test/`. Los pesos y datos grandes se excluyen de Git: para reproducir las cifras hay que conservar o recuperar esos artefactos por separado. El PDF de la rúbrica está fuera del repositorio en `C:\Users\alambre1\Downloads\Proyecto Final - Caso de uso real.pdf`.
