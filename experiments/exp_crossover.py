"""
Experimento: comparar operadores de crossover.

Varia el metodo de crossover manteniendo el resto de hiperparametros fijos.
Metodos: OnePoint, TwoPoint, Uniform, Annular, Aritmetico.
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
    ("OnePoint", {"crossover_methods": ["OnePoint"]}),
    ("TwoPoint", {"crossover_methods": ["TwoPoint"]}),
    ("Uniform", {"crossover_methods": ["Uniform"]}),
    ("Annular", {"crossover_methods": ["Annular"]}),
    ("Aritmetico", {"crossover_methods": ["Aritmetico"], "arithmetic_alpha": 0.5}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "crossover")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de operadores de crossover")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="crossover",
        configs=METHODS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
