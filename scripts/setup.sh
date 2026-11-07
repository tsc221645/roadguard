#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
profile="${1:-cpu}"
python_bin="${ROADGUARD_PYTHON:-python3.11}"
if [[ "$profile" != cpu && "$profile" != gpu ]]; then
  echo 'Uso: bash scripts/setup.sh cpu|gpu' >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11), "Se requiere Python 3.11"'
if [[ "$profile" == cpu ]]; then
  .venv/bin/python -m pip install -r requirements.cpu.lock.txt
else
  .venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "Instala primero PyTorch CUDA según docs/training_environments.md"'
  .venv/bin/python -m pip install -r requirements.app.lock.txt
fi
.venv/bin/python scripts/check_environment.py --output "artifacts/reports/environment_${profile}.json"
