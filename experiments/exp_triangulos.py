"""
Experimento: comparar cantidad de triangulos.

Varia triangle_count para observar el trade-off entre calidad de
aproximacion y costo computacional.
Valores: 10, 50, 200.

Observaciones esperadas:
- Mas triangulos = mejor aproximacion pero mayor tiempo por generacion.
- Existe un punto de rendimiento decreciente donde agregar triangulos
  no mejora significativamente el fitness.
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

# Cantidades de triangulos a comparar
VALUES = [10, 50, 200]

CONFIGS = [
    (f"triangulos_{v}", {"triangle_count": v})
    for v in VALUES
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "triangulos")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de cantidad de triangulos")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="triangulos",
        configs=CONFIGS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
