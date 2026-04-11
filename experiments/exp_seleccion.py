"""
Experimento: comparar metodos de seleccion.

Varia el metodo de seleccion manteniendo el resto de hiperparametros fijos.
Metodos: Elite, Ruleta, Universal, Ranking, Boltzmann,
TorneosDeterministicos y TorneosProbabilisticos.
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
    ("Elite", {"selection_method": "Elite"}),
    ("Ruleta", {"selection_method": "Ruleta"}),
    ("Universal", {"selection_method": "Universal"}),
    ("Ranking", {"selection_method": "Ranking"}),
    ("Boltzmann", {
        "selection_method": "Boltzmann",
        "boltzmann_t0": 100.0,
        "boltzmann_tc": 1.0,
        "boltzmann_k": 0.01,
    }),
    ("TorneosDeterministicos", {
        "selection_method": "TorneosDeterministicos",
        "tournament_m": 5,
    }),
    ("TorneosProbabilisticos", {
        "selection_method": "TorneosProbabilisticos",
        "tournament_threshold": 0.75,
    }),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "seleccion")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de metodos de seleccion")
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment_all_images(
        name="seleccion",
        configs=METHODS,
        output_base=output_base,
        seeds=seeds,
        max_generations=max_gen or 3000,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
