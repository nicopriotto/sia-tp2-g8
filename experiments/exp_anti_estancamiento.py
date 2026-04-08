"""
Experimento: anti-estancamiento.

Compara corridas con y sin anti-estancamiento (mutation boost + reinicio parcial)
en las 4 imagenes principales.
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
    ("Sin_anti_estancamiento", {
        "stagnation_check_interval": 0,
    }),
    ("Con_anti_estancamiento", {
        "stagnation_check_interval": 200,
        "stagnation_threshold": 0.0001,
        "stagnation_mutation_boost": 4.0,
        "stagnation_replace_pct": 0.2,
    }),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "anti_estancamiento")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de anti-estancamiento")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="anti_estancamiento",
        configs=CONFIGS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen or 2000,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
