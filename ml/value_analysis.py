"""Escenario ilustrativo de valor e intervalo bootstrap pareado por imagen."""
import argparse
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cost(row, missed_cost, review_cost):
    return row["fn"] * missed_cost + (row["tp"] + row["fp"]) * review_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, default=ROOT / "artifacts/evaluations/model_a_test/operational.json")
    parser.add_argument("--model-b", type=Path, default=ROOT / "artifacts/evaluations/model_b_test/operational.json")
    parser.add_argument("--missed-cost", type=float, default=100.0,
                        help="Supuesto en Q por daño etiquetado omitido")
    parser.add_argument("--review-cost", type=float, default=5.0,
                        help="Supuesto en Q por alerta revisada")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluations/value_scenario.json")
    args = parser.parse_args()
    a = json.loads(args.model_a.read_text(encoding="utf-8"))
    b = json.loads(args.model_b.read_text(encoding="utf-8"))
    if a["split"] != b["split"] or a["confidence"] != b["confidence"] or a["imgsz"] != b["imgsz"]:
        parser.error("Los reportes deben compartir split, confianza y resolución")
    by_image_a = {row["image"]: row for row in a["per_image"]}
    by_image_b = {row["image"]: row for row in b["per_image"]}
    if by_image_a.keys() != by_image_b.keys():
        parser.error("Los reportes no corresponden a las mismas imágenes")
    differences = [cost(by_image_a[name], args.missed_cost, args.review_cost) -
                   cost(by_image_b[name], args.missed_cost, args.review_cost)
                   for name in sorted(by_image_a)]
    n = len(differences)
    rng = random.Random(42)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n))
                     for _ in range(args.samples))
    report = {
        "scenario": "Ilustrativo; costos supuestos, no medidos",
        "currency": "Q", "missed_cost": args.missed_cost,
        "review_cost_per_alert": args.review_cost,
        "a_cost": cost(a["total"], args.missed_cost, args.review_cost),
        "b_cost": cost(b["total"], args.missed_cost, args.review_cost),
        "b_saving_vs_a": sum(differences),
        "paired_image_bootstrap_95_percent": [samples[int(0.025 * args.samples)],
                                              samples[int(0.975 * args.samples)]],
        "bootstrap_samples": args.samples, "images": n, "seed": 42,
        "sensitivity": [
            {"factor": factor,
             "b_saving_vs_a": cost(a["total"], args.missed_cost * factor, args.review_cost) -
                              cost(b["total"], args.missed_cost * factor, args.review_cost)}
            for factor in (0.5, 0.8, 1.0, 1.2, 1.5)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
