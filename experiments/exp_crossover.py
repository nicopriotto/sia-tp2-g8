"""
Experimento: comparar operadores de crossover.

Varia el metodo de crossover manteniendo el resto de hiperparametros fijos.
Metodos: OnePoint, TwoPoint, Uniform, Annular, Aritmetico.

Hipotesis:
- Uniform deberia funcionar mejor que OnePoint cuando la posicion del
  triangulo en el cromosoma no tiene significado semantico fuerte.
- Aritmetico genera hijos intermedios que pueden refinar soluciones.
"""
import os
import sys
import logging

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import (
    run_experiment,
    parse_common_args,
    apply_common_args,
)

# Operadores de crossover a comparar
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

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="crossover",
        configs=METHODS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
