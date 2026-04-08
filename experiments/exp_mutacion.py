"""
Experimento: comparar operadores de mutacion.

Varia el metodo de mutacion manteniendo el resto de hiperparametros fijos.
Metodos: Gen, MultiGen, Uniforme, Completa, NoUniforme, Gaussiana.
Corre en las 4 imagenes principales.
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

METHODS = [
    ("Gen", {"mutation_methods": ["Gen"]}),
    ("MultiGen", {"mutation_methods": ["MultiGen"]}),
    ("Uniforme", {"mutation_methods": ["Uniforme"]}),
    ("Completa", {"mutation_methods": ["Completa"]}),
    ("NoUniforme", {"mutation_methods": ["NoUniforme"], "non_uniform_b": 1.0}),
    ("Gaussiana", {"mutation_methods": ["Gaussiana"], "gaussian_sigma": 0.1}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "mutacion")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de operadores de mutacion")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="mutacion",
        configs=METHODS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
