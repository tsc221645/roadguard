"""Captura reproducible del entorno; no descarga pesos ni entrena modelos."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return {"available": False}
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    return {"available": True, "returncode": result.returncode,
            "output": (result.stdout + result.stderr).strip()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "fastapi", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"python": sys.version, "platform": platform.platform(),
              "packages": packages,
              "tools": {name: command(cmd) for name, cmd in {
                  "git": ["git", "--version"], "node": ["node", "--version"],
                  "npm": ["npm", "--version"], "ffmpeg": ["ffmpeg", "-version"],
                  "nvidia": ["nvidia-smi"],
              }.items()}}
    try:
        import torch
        report["torch"] = {"cuda_available": torch.cuda.is_available(),
                           "cuda_build": torch.version.cuda,
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "cpu_operation": (torch.ones(2) + torch.ones(2)).tolist()}
    except ImportError:
        report["torch"] = {"status": "not_installed"}
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
