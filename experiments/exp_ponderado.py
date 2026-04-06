"""
Experimento: Boltzmann(peso 0.7) + Elite(peso 0.3) vs cada uno solo.

Hipotesis: La combinacion ponderada permite exploracion temprana (Boltzmann)
y elitismo (Elite) simultaneamente, obteniendo mejor resultado que cualquiera solo.
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

# Configuraciones a comparar
CONFIGS = [
    (
        "Boltzmann70_Elite30",
        {
            "selection_methods": [
                {"method": "Boltzmann", "weight": 0.7},
                {"method": "Elite", "weight": 0.3},
            ],
            "boltzmann_t0": 100.0,
            "boltzmann_tc": 1.0,
            "boltzmann_k": 0.01,
        },
    ),
    (
        "Boltzmann_solo",
        {
            "selection_methods": ["Boltzmann"],
            "boltzmann_t0": 100.0,
            "boltzmann_tc": 1.0,
            "boltzmann_k": 0.01,
        },
    ),
    (
        "Elite_solo",
        {
            "selection_methods": ["Elite"],
        },
    ),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "ponderado")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de operadores ponderados: Boltzmann+Elite vs solos")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="ponderado",
        configs=CONFIGS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
