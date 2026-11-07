# RoadGuard AI

MVP académico CC3092 para detectar daños de pavimento en video, asociar GPS aproximado y comparar dos modelos ajustados mediante métricas técnicas y valor económico.

Estado: fase 1 del integrante A. Estructura, entorno y protocolo en preparación; todavía no hay modelos entrenados, API ni frontend ejecutables.

## Documentación

- [Especificación](ROADGUARD_AI_ESPECIFICACION_DESARROLLO(1).md)
- [Plan del integrante A](docs/PLAN_INTEGRANTE_A.md) y [plan del integrante B](docs/PLAN_INTEGRANTE_B.md)
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

Copiar `.env.example` a `.env` y ajustar rutas locales cuando se implementen los módulos de datos. No hay descarga automática de dataset ni entrenamiento en esta fase. `configs/classes.yaml` es la única correspondencia de clases.

Datos crudos, videos, GPX, pesos y secretos están excluidos de Git. Código bajo AGPL-3.0; datos y dependencias mantienen sus propias licencias. La nube es una opción de entrenamiento autorizada; no se han contratado recursos ni medido costos.
