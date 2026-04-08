"""
Experimento: funciones de fitness.

Compara LinearMSE, MSE, MAE y SSIM en las 4 imagenes principales.
Nota: SSIM es CPU-only y mas lenta.
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
    ("LinearMSE", {"fitness_function": "LinearMSE"}),
    ("MSE", {"fitness_function": "MSE"}),
    ("MAE", {"fitness_function": "MAE"}),
    ("SSIM", {"fitness_function": "SSIM"}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "fitness")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de funciones de fitness")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="fitness",
        configs=CONFIGS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen or 500,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
