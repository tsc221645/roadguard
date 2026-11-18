# Matriz de cumplimiento — Laboratorio 10

Estado de la ejecución del 17-11-2026. `results.json` y el PDF se encuentran en `explainability/outputs/`. «Cumplido» se refiere a la evidencia disponible, no a una garantía de validez causal.

| Requisito | Estado | Evidencia |
|---|---|---|
| Investigación XAI | CUMPLIDO | Tabla de seis métodos y fuentes en `explainability/README.md`; PDF p. 1. |
| Interpretabilidad vs. XAI | CUMPLIDO | Discusión de reglas de bordes frente a detector en README; PDF p. 1. |
| Propiedades formales | CUMPLIDO | Definición de `S`, atribución firmada y ausencia de completitud/causalidad en README. |
| Parámetros de diseño | CUMPLIDO | `results.json:config` y sección de método del README. |
| Método principal | CUMPLIDO | `run_xai.py:explain`; 25 mapas numéricos en `results.json`. |
| Alternativa descartada | CUMPLIDO | Comparación con Grad-CAM en README y PDF p. 1. |
| Modelo final | CUMPLIDO | `checkpoint` y SHA-256 en `results.json`; modelo B en README principal. |
| Sin reentrenamiento | CUMPLIDO | `run_xai.py` solo llama `YOLO.predict`; `predictions_unchanged=true`. |
| Cinco instancias | CUMPLIDO | Cinco IDs distintos en `results.json:local`; `local_cases.png`. |
| Alta confianza | CUMPLIDO | D00 TP 0.8691, IoU 0.721 en `results.json:local[0]`. |
| Error | CUMPLIDO | D00 FP 0.7142 en `results.json:local[1]`, según GT del test. |
| Caso cercano al umbral | CUMPLIDO | D20 0.1002 frente a conf 0.10 en `results.json:local[2]`. |
| Subgrupo | CUMPLIDO | D40 TP en `results.json:local[3]`; no hay etiquetas de condición visual. |
| Explicaciones locales | CUMPLIDO | `local_cases.png`, valores firmados y análisis en README y PDF p. 2. |
| Explicación global | CUMPLIDO | 20 imágenes adicionales, `global_summary.png`, `results.json:global`; PDF p. 3. |
| Reproducibilidad | CUMPLIDO | Comando ejecutado, semilla, versiones, hash y configuración en README y JSON. |
| Interpretación de dominio | CUMPLIDO | Análisis de grietas, superficie y caja D40 en README. |
| Atajos y proxies | PARCIAL | Revisión visual de cinco casos; no hay auditoría sistemática por cámara, luz o sombras. |
| Análisis de errores | CUMPLIDO | FP D00 y D20, FN sin caja, limitación de anotaciones en README. |
| Decisiones concretas | CUMPLIDO | Revisión de FP/FN, auditoría de GT y validación previa a cambios en README. |
| Audiencia objetivo | CUMPLIDO | Propuesta de caja, confianza, mapa opcional y validación humana en README. |
| Modelo interpretable en alto riesgo | CUMPLIDO | Discusión de comparación justa pendiente en README y PDF. |
| Revisión de la elección | CUMPLIDO | Costo observado y limitación de Grad-CAM en README y PDF p. 4. |
| Información adicional a métricas | CUMPLIDO | Comparación entre sensibilidad local y mAP/recall en README y PDF p. 4. |
| Costo computacional | CUMPLIDO | `results.json:benchmark`, hardware/versión, tiempos y método en README. |
| Viabilidad productiva | CUMPLIDO | Discusión de tiempo real frente a lotes/auditorías en README y PDF p. 4. |
| Riesgos éticos | CUMPLIDO | Riesgos de FN, FP, privacidad, GPS y uso indebido en README/PDF p. 4. |
| PDF máximo 4 páginas | CUMPLIDO | `laboratorio10_xai.pdf`; conteo automático de cuatro páginas. |
| Repositorio | CUMPLIDO | Código, resultados, figuras y PDF regenerables en `explainability/`; sin dataset ni checkpoint duplicado. |
| README | CUMPLIDO | Sección «Laboratorio 10 — Explainable AI (XAI)» del README principal. |
| Enlace al repositorio | CUMPLIDO | Remoto `https://github.com/tsc221645/roadguard.git`, al final del PDF. |

Pendiente de investigación adicional, fuera de este laboratorio: evaluación con subgrupos documentados de iluminación/cámara y revisión manual de la naturaleza física de los FP y FN.
