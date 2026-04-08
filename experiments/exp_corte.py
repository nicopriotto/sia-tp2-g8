"""
Experimento: comparar criterios de corte.

Compara distintas condiciones de terminacion en las 4 imagenes principales.
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
    ("solo_generaciones", {
        "content_generations": 0,
        "content_threshold": 0.0,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    ("contenido_50gen", {
        "content_generations": 50,
        "content_threshold": 0.001,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    ("contenido_100gen", {
        "content_generations": 100,
        "content_threshold": 0.001,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    ("estructura_30gen", {
        "content_generations": 0,
        "content_threshold": 0.0,
        "structure_generations": 30,
        "structure_threshold": 0.001,
    }),
    ("combinado", {
        "content_generations": 50,
        "content_threshold": 0.001,
        "structure_generations": 30,
        "structure_threshold": 0.001,
    }),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "corte")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de criterios de corte")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="criterios_corte",
        configs=CONFIGS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
