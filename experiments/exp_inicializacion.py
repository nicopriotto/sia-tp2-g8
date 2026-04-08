"""
Experimento: inicializacion de la poblacion.

Compara random puro vs smart_init en las 4 imagenes principales.
"""
import os
import sys
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import (
    run_experiment_all_images,
    parse_common_args,
    apply_common_args,
)

CONFIGS = [
    ("Random", {"smart_init": False}),
    ("SmartInit", {"smart_init": True}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "inicializacion")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de inicializacion de poblacion")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="inicializacion",
        configs=CONFIGS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen or 500,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
