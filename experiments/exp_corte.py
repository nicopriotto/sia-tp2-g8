"""
Experimento: comparar criterios de corte.

Compara distintas condiciones de terminacion:
- Solo max_generations (baseline)
- Contenido: fitness no mejora en N generaciones
- Estructura: diversidad cae por debajo de umbral

Observaciones esperadas:
- Contenido detiene la ejecucion cuando el GA se estanca, ahorrando tiempo.
- Estructura detecta convergencia prematura por baja diversidad.
- La combinacion de criterios permite balancear calidad y eficiencia.
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

# Criterios de corte a comparar
CONFIGS = [
    # Baseline: solo generaciones maximas
    ("solo_generaciones", {
        "content_generations": 0,
        "content_threshold": 0.0,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    # Contenido: detener si no mejora en 50 generaciones (delta < 0.001)
    ("contenido_50gen", {
        "content_generations": 50,
        "content_threshold": 0.001,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    # Contenido: detener si no mejora en 100 generaciones
    ("contenido_100gen", {
        "content_generations": 100,
        "content_threshold": 0.001,
        "structure_generations": 0,
        "structure_threshold": 0.0,
    }),
    # Estructura: detener si std < 0.001 por 30 generaciones
    ("estructura_30gen", {
        "content_generations": 0,
        "content_threshold": 0.0,
        "structure_generations": 30,
        "structure_threshold": 0.001,
    }),
    # Combinado: contenido + estructura
    ("combinado", {
        "content_generations": 50,
        "content_threshold": 0.001,
        "structure_generations": 30,
        "structure_threshold": 0.001,
    }),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "corte")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de comparacion de criterios de corte")
    args = parser.parse_args()

    max_gen, seeds, image_path = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    run_experiment(
        name="criterios_corte",
        configs=CONFIGS,
        output_base=output_base,
        image_path=image_path,
        seeds=seeds,
        max_generations=max_gen,
    )

    print(f"\nResultados guardados en: {output_base}")


if __name__ == "__main__":
    main()
