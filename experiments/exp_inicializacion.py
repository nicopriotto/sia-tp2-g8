"""
Experimento: inicializacion de la poblacion.

Compara random puro vs smart_init en las 4 imagenes principales.
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
from experiments.plot_cross_image import (
    load_cross_image_data,
    _natural_sort_key,
)

CONFIGS = [
    ("Random", {"smart_init": False}),
    ("SmartInit", {"smart_init": True}),
]

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "inicializacion")
DEFAULT_HEATMAP_OUTPUT = os.path.join(
    PROJECT_ROOT,
    "experiments",
    "plots",
    "inicializacion",
    "inicializacion_heatmap_fitness.png",
)


def regenerate_gen0_heatmap(results_dir: str, output_path: str):
    """Regenera el heatmap cruzado usando fitness en generacion 0."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        logging.error("No se pudo importar matplotlib/seaborn para plotear: %s", exc)
        return

    data = load_cross_image_data(results_dir)
    if data.empty:
        logging.warning("Sin datos para regenerar heatmap de inicializacion en: %s", results_dir)
        return

    pivot = data.groupby(["config", "imagen"])["fitness_gen0"].mean().unstack()
    pivot = pivot.loc[sorted(pivot.index, key=_natural_sort_key)]

    plt.figure(figsize=(max(8, len(pivot.columns) * 2.5), max(4, len(pivot) * 0.6)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        linewidths=0.5,
        vmin=pivot.values.min() * 0.98,
        vmax=min(1.0, pivot.values.max() * 1.005),
        cbar_kws={"label": "Fitness gen 0"},
    )
    plt.title("Fitness inicial (gen 0) por configuracion e imagen\ninicializacion", fontsize=13)
    plt.xlabel("Imagen", fontsize=11)
    plt.ylabel("Configuracion", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info("Heatmap gen 0 guardado: %s", output_path)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    parser = parse_common_args("Experimento de inicializacion de poblacion")
    parser.add_argument(
        "--plot_only", action="store_true",
        help="Solo regenera el heatmap cruzado usando fitness en gen 0"
    )
    parser.add_argument(
        "--heatmap_output", type=str, default=DEFAULT_HEATMAP_OUTPUT,
        help="Ruta de salida para el heatmap cruzado de inicializacion"
    )
    args = parser.parse_args()

    max_gen, seeds, _ = apply_common_args(args)
    output_base = args.output or DEFAULT_OUTPUT

    if not args.plot_only:
        run_experiment_all_images(
            name="inicializacion",
            configs=CONFIGS,
            output_base=output_base,
            seeds=seeds,
            max_generations=max_gen or 500,
        )

        print(f"\nResultados guardados en: {output_base}")

    regenerate_gen0_heatmap(output_base, args.heatmap_output)


if __name__ == "__main__":
    main()
