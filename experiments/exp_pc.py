"""
Experimento: comparar probabilidades de crossover.

Varia crossover_probability (Pc) para evaluar su efecto en la convergencia.
Valores: 0.5, 0.7, 0.9, 1.0.

Observaciones esperadas:
- Pc muy bajo reduce la combinacion de material genetico.
- Pc = 1.0 maximiza la mezcla pero puede destruir buenos esquemas.
- Valores intermedios (0.7-0.9) suelen ser optimos.
"""
import os
import sys
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import (
    run_experiment,
    parse_common_args,
    apply_common_args,
)

# Probabilidades de crossover a comparar
VALUES = [0.5, 0.7, 0.9, 1.0]

CONFIGS = [
    (f"pc_{v}", {"crossover_probability": v})
    for v in VALUES
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "pc")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de probabilidad de crossover")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="probabilidad_crossover",
        configs=CONFIGS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
