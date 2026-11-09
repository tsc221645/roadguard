# RoadGuard AI

MVP académico CC3092 para detectar daños de pavimento en video, asociar GPS aproximado y comparar dos modelos ajustados mediante métricas técnicas y valor económico.

Estado: preparación del dataset y del pipeline. Todavía no hay modelos entrenados, API ni frontend ejecutables.

## Documentación

- [Especificación](ROADGUARD_AI_ESPECIFICACION_DESARROLLO(1).md)
- [Informe incremental, secciones 4.1–4.9](docs/INFORME.md)
- [Arquitectura](docs/architecture.md), [protocolo experimental](docs/experimental_protocol.md)
- [Entornos CPU, RTX 3050 y nube](docs/training_environments.md)
- [Ética, privacidad y licencias](docs/ethics_privacy.md)

## Entorno local

Python 3.11; no usar el Python global sin comprobar su versión. Consultar la guía de entornos para la instalación CPU o CUDA.

```bash
source .venv/bin/activate
python scripts/check_environment.py --output artifacts/reports/environment_local.json
```

Para entrenamiento en Windows con la RTX 3050, usar el entorno Conda `dm1`:

```powershell
$env:PYTHONNOUSERSITE = "1"
conda run -n dm1 python -m ml.train --name model_a --epochs 25
conda run -n dm1 python -m ml.train --name model_b --epochs 25 --lr0 0.0005 --degrees 3
```

Copiar `.env.example` a `.env` y ajustar rutas locales para los módulos de datos. El dataset se descarga con `scripts/download_rdd.py`; el smoke test ya fue validado y el entrenamiento completo A/B está pendiente. `configs/classes.yaml` es la única correspondencia de clases.

Datos crudos, videos, GPX, pesos y secretos están excluidos de Git. Código bajo AGPL-3.0; datos y dependencias mantienen sus propias licencias. La nube es una opción de entrenamiento autorizada; no se han contratado recursos ni medido costos.
