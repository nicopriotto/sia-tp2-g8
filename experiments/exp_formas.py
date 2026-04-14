"""
Experimento: triangulos vs elipses.

Compara gene_type="triangle" vs "ellipse" en 3 imagenes para presentacion:
- 1_Ucrania (images/1.jpg)
- 2_Apple (images/3.jpg)
- 3_Cubista (images/4.jpg)
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
    ("Triangulo", {"gene_type": "triangle"}),
    ("Elipse", {"gene_type": "ellipse"}),
]

IMAGES = [
    ("1_Ucrania", os.path.join(PROJECT_ROOT, "images", "1.jpg")),
    ("2_Apple", os.path.join(PROJECT_ROOT, "images", "3.jpg")),
    ("3_Cubista", os.path.join(PROJECT_ROOT, "images", "4.jpg")),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "formas")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de triangulos vs elipses")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="formas",
        configs=CONFIGS,
        output_base=output_base,
        images=IMAGES,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
