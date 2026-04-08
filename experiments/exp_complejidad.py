"""
Experimento: complejidad de imagenes.

Corre la misma configuracion en las 4 imagenes principales para mostrar
como la complejidad visual afecta la velocidad de convergencia.

Imagenes:
  1.jpg - Bandera Ucrania (2 colores, muy simple)
  2.jpg - Bandera Grecia (patron geometrico)
  3.jpg - Logo Apple (silueta con fondo)
  4.png - Pintura cubista (muchos colores y formas)
"""
import os
import sys
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import (
    BASE_CONFIG,
    SEEDS,
    run_single,
    parse_common_args,
    apply_common_args,
)

IMAGES = [
    ("1_Ucrania", os.path.join(PROJECT_ROOT, "images", "1.jpg")),
    ("2_Grecia", os.path.join(PROJECT_ROOT, "images", "2.jpg")),
    ("3_Apple", os.path.join(PROJECT_ROOT, "images", "3.jpg")),
    ("4_Cubista", os.path.join(PROJECT_ROOT, "images", "4.png")),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "complejidad")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de complejidad de imagenes")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    seeds = seeds or SEEDS
    output_base = args.output or DEFAULT_OUTPUT

    config_dict = {**BASE_CONFIG}
    if max_gen is not None:
        config_dict["max_generations"] = max_gen

    for label, image_path in IMAGES:
        for seed in seeds:
            run_dir = os.path.join(output_base, label, f"seed_{seed}")
            print(f"[complejidad] {label}, seed={seed} -> {run_dir}")
            try:
                run_single(config_dict, image_path, run_dir, seed)
            except Exception as e:
                print(f"  ERROR: {e}")

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
