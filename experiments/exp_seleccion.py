"""
Experimento: comparar metodos de seleccion.

Varia el metodo de seleccion manteniendo el resto de hiperparametros fijos.
Metodos: Elite, Ruleta, Universal, Ranking, Boltzmann, TorneosDeterministicos.

Hipotesis:
- Ruleta con fitness dispares puede converger prematuramente.
- Ranking normaliza y resuelve el problema de fitness dispares.
- Boltzmann permite exploracion temprana y explotacion tardia.
- Torneos es robusto y facil de parametrizar.
"""
import os
import sys
import logging

# Agregar directorio raiz al path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import (
    run_experiment,
    parse_common_args,
    apply_common_args,
)

# Metodos de seleccion a comparar con sus parametros especificos
METHODS = [
    ("Elite", {"selection_method": "Elite"}),
    ("Ruleta", {"selection_method": "Ruleta"}),
    ("Universal", {"selection_method": "Universal"}),
    ("Ranking", {"selection_method": "Ranking"}),
    ("Boltzmann", {"selection_method": "Boltzmann", "boltzmann_t0": 100.0, "boltzmann_tc": 1.0, "boltzmann_k": 0.01}),
    ("TorneosDeterministicos", {"selection_method": "TorneosDeterministicos", "tournament_m": 5}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "seleccion")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de metodos de seleccion")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="seleccion",
        configs=METHODS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
