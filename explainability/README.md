# Laboratorio 10 — Explainable AI en RoadGuard

## Evidencia y alcance

El repositorio remoto es `https://github.com/tsc221645/roadguard.git`. La aplicación usa por defecto el modelo B (`YOLO11n`) de `artifacts/runs/model_b/weights/best.pt`. Este laboratorio lo carga sin reentrenar y sin alterar sus pesos. El SHA-256 del checkpoint usado el 17 de noviembre de 2026 fue `a690a77b543dc9a4bb0e9065c18e809c8df1abd8b873c9f59e04fab2ac83c03e`. Las clases de `configs/classes.yaml` y del checkpoint coinciden: `0=D00` grieta longitudinal, `1=D10` grieta transversal, `2=D20` piel de cocodrilo, `3=D40` bache.

Se usaron las 1 072 imágenes del test Czech de RDD2022 y sus etiquetas YOLO en `data/processed/rdd2022/{images,labels}/test`. Las particiones son train, val y test por paquete de origen; no se usaron imágenes de Guatemala para este laboratorio. La inferencia usa `conf=0.10`, `imgsz=640`, NMS IoU 0.50 (valor efectivo de Ultralytics), y emparejamiento con GT de la misma clase a IoU ≥0.50. El inventario reprodujo **533 VP, 809 FP y 1 212 FN** del reporte operativo B. Un FP significa que la caja no coincide con una anotación del test bajo este protocolo: no confirma por sí mismo ausencia física de daño. Los FN no poseen una detección objetivo que pueda explicarse con esta técnica.

## Investigación y elección del método

| Método | Idea y adaptación a YOLO | Gradientes / caja negra | Requisitos y costo orientativo | Interpretación y límite |
|---|---|---|---|---|
| Saliency | Derivada de un objetivo escalar por caja respecto a píxeles. | Sí / no | Una retropropagación; objetivo diferenciable previo a NMS. | Sensibilidad local, a menudo ruidosa; la formulación de clasificación no identifica automáticamente una caja. |
| SmoothGrad | Promedio de saliency sobre entradas con ruido. | Sí / no | Varias retropropagaciones y amplitud de ruido. | Reduce ruido visual, pero depende del objetivo y del ruido elegido. |
| Integrated Gradients | Integra gradientes desde una línea base hasta la imagen. | Sí / no | Línea base y decenas de pasos con gradientes. | Completitud respecto al objetivo escalar y baseline definidos; no garantiza explicación de la caja final tras NMS. |
| Grad-CAM | Pondera activaciones de una capa con gradientes del objetivo. | Sí / no | Capa convolucional y objetivo diferenciable; una pasada hacia atrás. | Mapa grueso; asociar la caja final a un objetivo anterior a NMS exige validación. |
| Oclusión | Oculta regiones y mide cómo cambia una detección emparejada por clase/IoU. | No / sí | Rejilla, reemplazo y 64 inferencias por mapa 8×8. | Sensibilidad de la salida final; alto costo, artefactos de ocultación. **Método elegido.** |
| Conceptos/TCAV | Relaciona activaciones con direcciones de conceptos visuales. | Sí / no | Ejemplos anotados de conceptos, capa y objetivo; extracción y gradientes adicionales. | Aporta lenguaje de dominio, pero depende de conceptos bien definidos y de su procedencia. |

Estos métodos se formularon principalmente para clasificación; en detección se necesita fijar clase y caja objetivo. Se eligió oclusión frente a Grad-CAM porque opera sobre las detecciones finales de este checkpoint sin derivar a través del posprocesamiento. El costo medido fue considerable, por lo que la elección sigue siendo adecuada para auditoría y análisis de errores, no para explicar cada alerta en tiempo real. No se implementó Grad-CAM y no se le atribuyen resultados. Un sistema interpretable por diseño, por ejemplo reglas de bordes y textura, mostraría directamente su lógica, pero no es comparable como detector multiclase sin construirlo y evaluarlo en otro experimento; aquí está prohibido reentrenar.

Como herramientas de Python, [Captum](https://captum.ai/docs/attribution_algorithms) ofrece Saliency, Integrated Gradients, NoiseTunnel (para SmoothGrad) y oclusión sobre una función objetivo definida por el usuario; los métodos de gradientes exigen que esa función sea diferenciable. [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) facilita mapas CAM, pero también requiere fijar correctamente el objetivo de la detección. Este laboratorio implementa la oclusión directamente con Ultralytics y OpenCV para medir la salida final emparejada por caja y clase. Ninguna de esas bibliotecas XAI adicionales es una dependencia del experimento ejecutado.

La oclusión **no** tiene la completitud de Integrated Gradients ni garantías de causalidad, aditividad o fidelidad universal. Una región puede afectar tanto la confianza como la localización. `S(x)` combina ambas: máximo de `confianza × IoU` entre cajas de la misma clase con IoU ≥0.50 respecto a la caja original. Si no hay correspondencia, `S=0`. El inventario calcula y guarda `S(original)` para cada detección; puede superar su confianza si otra caja superpuesta de la misma clase logra mayor producto. En los 25 casos seleccionados coincidió con la confianza original. La atribución firmada por celda es `S(original)−S(ocluida)`; valores negativos significan que ocultar esa celda aumentó la puntuación. `results.json` conserva ambos signos; el mapa visual muestra solo caídas positivas, normalizadas **por caso**, por lo que los colores entre casos no son comparables en magnitud absoluta.

La imagen se divide en 8×8 celdas no solapadas; cada celda se sustituye por la misma región de la imagen desenfocada con Gaussian blur 51×51. La selección, aleatoriedad y prueba de fidelidad usan semilla 42. Cambiar rejilla, desplazamiento, reemplazo, tamaño de imagen, confianza, NMS o IoU puede cambiar mapa y selección. No se estudió sistemáticamente esa sensibilidad. Cerca de `conf=0.10`, una pequeña caída puede eliminar la detección del posprocesamiento y llevar `S` a cero: el mapa incluye ese salto de umbral. Una capa objetivo no aplica a oclusión; sí sería crítica en Grad-CAM. El desenfoque puede generar entradas fuera de la distribución de entrenamiento. El mapa no revela parámetros internos ni prueba que el modelo aprendió la característica que aparenta resaltar.

Fuentes: [Zeiler y Fergus, visualización y oclusión](https://papers.baulab.info/papers/Zeiler-2014.pdf); [Selvaraju et al., Grad-CAM](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf); [Smilkov et al., SmoothGrad](https://arxiv.org/abs/1706.03825); [Sundararajan et al., Integrated Gradients](https://arxiv.org/abs/1703.01365); [Kim et al., TCAV](https://arxiv.org/abs/1711.11279).

## Procedimiento reproducible

Con `dm1` activo, PyTorch y las dependencias de `requirements.in` instaladas, ejecutar desde la raíz:

```powershell
$env:PYTHONNOUSERSITE = "1"
python -m explainability.run_xai --device 0 --grid 8 --batch 16
```

Para CPU usar `--device cpu`; el lote se puede bajar con `--batch 4`. La prueba de integración `--smoke --output explainability/outputs_smoke` recorre el mismo test, explica cinco casos locales y uno global, y escribe solo JSON. El proceso completo produce `explainability/outputs/results.json`, `local_cases.png`, `global_summary.png` y `laboratorio10_xai.pdf`. Las figuras, tablas y PDF se regeneran desde las predicciones reales. Las imágenes y etiquetas del test, excluidas de Git, se preparan como indica el README principal. El checkpoint original ya existe en el repositorio; no se crea una copia. El comando es idempotente respecto a los datos y pesos, aunque sobrescribe sus propios artefactos regenerables.

El script valida modelo/clases, inventaría el test, selecciona casos antes de ver mapas y registra IDs, cajas, estado, IoU, parámetros, versión, hardware, fecha y hash del checkpoint. Selección local: TP de mayor confianza; FP de mayor confianza; detección más próxima a 0.10; TP D40 de mayor confianza; TP D10 de mayor confianza disponible. Exige cinco imágenes distintas. Para el análisis global, excluye esos casos y elige cinco detecciones por clase en orden de hash SHA-256 de semilla e ID de imagen. Son **20 imágenes adicionales**, estratificadas por clase predicha, no una muestra aleatoria representativa de todo el test. En el grupo global hay 6 TP y 14 FP. La correspondencia con GT es uno a uno, ordenada por confianza, como `ml.operational_metrics`.

## Resultados locales del 17-11-2026

| Caso | ID de imagen | Clase y estado | Confianza | Observación acotada |
|---|---|---:|---:|---|
| TP alta | `9f5b490e9336a749e1cbf2d4.jpg` | D00 TP | 0.8691 | La atribución positiva se concentra alrededor de la grieta anotada; IoU con GT 0.721. No demuestra reconocimiento causal de bordes. |
| Error | `cf83bcd6de724e0f8fc43f41.jpg` | D00 FP | 0.7142 | La puntuación depende de una región de superficie vial; la anotación no respalda la caja. Junta, grieta sin anotar o textura similar requieren revisión humana. |
| Umbral | `b7cda0bd90ea98273647cea0.jpg` | D20 FP | 0.1002 | Distancia a 0.10: 0.0002. La importancia positiva se reparte más entre regiones y hay sensibilidad a perturbaciones fuera de la caja. |
| Subgrupo | `bf2c7bd27c5da22b9c24a0a4.jpg` | D40 TP | 0.5009 | El bache predicho es muy pequeño en la imagen; la rejilla 8×8 limita la precisión espacial. No hay etiquetas de iluminación o cámara para formar otro subgrupo. |
| Otra clase | `9906b1ba35c832e419638b5a.jpg` | D10 TP | 0.6216 | La sensibilidad se agrupa cerca de la discontinuidad anotada; el mapa no distingue por sí solo grieta de marca. |

En la figura, cian es predicción, verde discontinuo es GT y la escala `magma` (oscuro a amarillo) representa la caída positiva de `S`. Es una lectura visual de cinco casos seleccionados por reglas fijas. Se inspeccionaron posibles confusores como juntas, marcas, sombras y otras texturas; la muestra no permite confirmar un atajo sistemático, ni se midieron subgrupos de iluminación, cámara o país distintos de Czech. Un FP puede deberse tanto a error del modelo como a anotación incompleta. El FN no tiene caja predicha y no se inventó un mapa para él. Los valores firmados y las cajas de todos los casos están en `results.json`.

La revisión visual de esos cinco casos no identificó una marca de agua que explicara las detecciones. No se ejecutó una auditoría formal de fugas entre particiones ni una comparación controlada de cámaras o iluminación; un mapa de oclusión por sí solo tampoco puede descartar esas fugas. En los TP D00 y D10, la sensibilidad cerca de las anotaciones es compatible con la expectativa de que el detector atienda discontinuidades del pavimento. Los FP y la respuesta dispersa del caso D20 muestran que esa expectativa no se cumple de forma uniforme. Las métricas agregadas de detección no muestran esta variación por caja, pero los mapas tampoco establecen qué rasgo semántico aprendió la red.

## Resultados globales y fidelidad

| Clase | n | TP/FP | Media de caída máxima de S | Fracción positiva dentro de caja |
|---|---:|---:|---:|---:|
| D00 | 5 | 2/3 | 0.329 | 0.391 |
| D10 | 5 | 1/4 | 0.255 | 0.149 |
| D20 | 5 | 1/4 | 0.162 | 0.315 |
| D40 | 5 | 2/3 | 0.198 | 0.085 |

La fracción dentro pondera cada celda por su intersección con la caja predicha; cajas pequeñas como D40 pueden tener fracción baja incluso con una razón dentro/fuera alta. Estas estadísticas describen **detecciones seleccionadas**, no todas las imágenes ni la calidad diagnóstica. El gráfico compara caída máxima y distribución de importancia media dentro/fuera. La presencia de FP en todas las clases y la dispersión de algunos mapas justifican revisar anotaciones y casos difíciles antes de plantear cambios de datos o umbral. Mayor diversidad de carreteras y nuevas condiciones visuales son hipótesis de mejora, todavía no probadas por este experimento.

Como comprobación limitada de fidelidad, se ocultaron **juntas** cuatro celdas de mayor atribución, cuatro de menor atribución y cuatro aleatorias, usando la misma `S`. En 5/5 casos locales la primera opción redujo más `S` que las otras dos. Se hizo una sola selección aleatoria por caso; esto confirma consistencia con estas perturbaciones, no fidelidad universal ni causalidad. En `results.json` se guardan índices, puntuaciones y cambios. La inferencia normal del mismo fotograma antes y después de XAI produjo las mismas cajas, clases y confianzas redondeadas; también se comparó el SHA-256 del checkpoint cargado.

## Costo, decisiones y uso responsable

Medido en Windows 11, RTX 3050, Python 3.13.2, PyTorch 2.14.0+cu126 y Ultralytics 8.4.208, con calentamiento y sincronización CUDA: inferencia normal mediana de tres repeticiones **0.01146 s**; explicación local media de cinco **0.526 s**, aproximadamente **46×** la inferencia individual; cinco explicaciones **2.631 s**; muestra global de veinte **10.962 s**. Cada explicación hizo 64 inferencias perturbadas en lotes de 16; la fidelidad añadió tres por caso. Pico asignado de GPU: **419 MiB**. Estos tiempos miden llamadas a inferencia y oclusión, no lectura completa del dataset, revisión humana o generación del PDF; la comparación normal y XAI usa imágenes distintas de la misma resolución de entrada. Solo la inferencia normal se repitió tres veces; el tiempo por mapa es una medición por cada caso. CPU, otros tamaños y más detecciones pueden variar ampliamente.

Para la aplicación, mostrar caja, clase, confianza, imagen y mapa opcional a quien revisa alertas, con opción de confirmar o rechazar. El análisis sugiere priorizar FP ambiguos y revisar FN peligrosos con personas; antes de cambiar umbrales se requeriría medir el efecto sobre validación, y antes de ampliar datos, auditar anotaciones. El costo observado favorece auditorías periódicas y análisis por lotes selectivos; no respalda explicaciones de cada predicción en tiempo real. No usar estas salidas solas para cerrar calles, descartar daños o asignar presupuesto. La transferencia Czech→Guatemala, lluvia, poca luz, placas, rostros y GPS sensible requieren validación y controles adicionales. Una explicación visual no certifica una detección y puede inducir confianza excesiva. Si un sistema interpretable alcanzara rendimiento similar bajo evaluación justa, podría preferirse en decisiones de mayor riesgo; aquí no se ejecutó esa comparación.

Decisión inmediata sustentada por el caso FP D00: pedir revisión humana antes de registrar una incidencia como confirmada. Hipótesis por verificar: revisar si la anotación omitió una grieta o si la caja responde a una junta. Decisión sustentada por el caso D20 de 0.1002: mostrar la confianza y permitir filtrar alertas próximas al umbral; cambiar el umbral requiere evaluar VP, FP y FN en validación. Decisión sustentada por el costo medido: generar mapas solo al auditar errores o lotes seleccionados. Ninguna de estas acciones implica que el mapa sea una prueba de la clase correcta.

La matriz de evidencia y el estado de cada requisito está en [COMPLIANCE.md](COMPLIANCE.md).
