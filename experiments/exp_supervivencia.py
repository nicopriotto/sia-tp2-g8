"""
Experimento: comparar estrategias de supervivencia.

Compara Aditiva (padres + hijos compiten) vs Exclusiva (solo hijos reemplazan).

Hipotesis:
- Aditiva preserva mejor las buenas soluciones (elitismo implicito)
  pero puede reducir diversidad.
- Exclusiva fuerza renovacion total, mayor diversidad pero riesgo de
  perder buenas soluciones.
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

# Estrategias de supervivencia a comparar
METHODS = [
    ("Aditiva", {"survival_strategy": "Aditiva"}),
    ("Exclusiva", {"survival_strategy": "Exclusiva"}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "supervivencia")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de estrategias de supervivencia")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="supervivencia",
        configs=METHODS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
