"""
Genera graficos de analisis cruzado entre imagenes para un experimento.

Recibe un directorio de resultados con sub-carpetas por imagen:
    results/<exp>/1_Ucrania/<configs>/seeds/
    results/<exp>/2_Grecia/<configs>/seeds/
    ...

Genera:
- Heatmap: config x imagen -> fitness final
- Barras agrupadas: fitness final por config, agrupado por imagen
- Heatmap de convergencia: gen para llegar a 90% del fitness final
- Tabla resumen CSV

Uso:
    python plot_cross_image.py --input experiments/results/seleccion --output experiments/plots/seleccion
    python plot_cross_image.py --all --input experiments/results --output experiments/plots
"""
import argparse
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import seaborn as sns
except ImportError as e:
    print(f"Error: dependencia no instalada: {e}")
    print("Instalar con: pip install matplotlib pandas seaborn")
    sys.exit(1)


def _natural_sort_key(name: str):
    """Ordena de forma logica: numeros por valor, texto alfabetico."""
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _pretty_config_label(name: str) -> str:
    """Acorta labels largas para mejorar legibilidad en ejes."""
    if name.endswith("_triangulos"):
        return name.replace("_triangulos", "")
    return name


def _parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_image_renames(raw: str) -> dict[str, str]:
    """
    Parsea mapping estilo: src1:dst1,src2:dst2.
    """
    mapping: dict[str, str] = {}
    if not raw:
        return mapping

    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if ":" not in pair:
            print(f"  Advertencia: mapping invalido '{pair}', se ignora")
            continue
        src, dst = pair.split(":", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            print(f"  Advertencia: mapping invalido '{pair}', se ignora")
            continue
        mapping[src] = dst
    return mapping


def load_cross_image_data(
    results_dir: str,
    include_images: set[str] | None = None,
    image_renames: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Carga datos de todas las imagenes y configs en un DataFrame unificado.

    Retorna DataFrame con columnas:
        imagen, config, seed, fitness_final, gen_total, gen_90pct,
        fitness_std_final, avg_fitness_final, elapsed_seconds
    """
    rows = []

    if not os.path.isdir(results_dir):
        return pd.DataFrame()

    for img_dir in sorted(os.listdir(results_dir), key=_natural_sort_key):
        if include_images and img_dir not in include_images:
            continue

        img_path = os.path.join(results_dir, img_dir)
        if not os.path.isdir(img_path):
            continue

        img_label = image_renames.get(img_dir, img_dir) if image_renames else img_dir

        for config_dir in sorted(os.listdir(img_path), key=_natural_sort_key):
            config_path = os.path.join(img_path, config_dir)
            if not os.path.isdir(config_path):
                continue

            for seed_dir in sorted(os.listdir(config_path), key=_natural_sort_key):
                seed_path = os.path.join(config_path, seed_dir)
                csv_path = os.path.join(seed_path, "metrics.csv")
                if not os.path.exists(csv_path):
                    continue

                try:
                    df = pd.read_csv(csv_path)
                    if len(df) == 0:
                        continue

                    final_fitness = df["best_fitness"].iloc[-1]
                    final_avg = df["avg_fitness"].iloc[-1]
                    final_std = df["fitness_std"].iloc[-1]
                    total_gens = df["generation"].iloc[-1]
                    elapsed = df["elapsed_seconds"].iloc[-1]

                    # Gen para 90% del fitness final
                    target_90 = final_fitness * 0.9
                    reached = df[df["best_fitness"] >= target_90]
                    gen_90 = reached["generation"].iloc[0] if len(reached) > 0 else total_gens

                    # Gen para 95%
                    target_95 = final_fitness * 0.95
                    reached_95 = df[df["best_fitness"] >= target_95]
                    gen_95 = reached_95["generation"].iloc[0] if len(reached_95) > 0 else total_gens

                    seed = int(seed_dir.replace("seed_", ""))

                    # Fitness en gen 0
                    fitness_gen0 = df["best_fitness"].iloc[0]

                    rows.append(
                        {
                            "imagen": img_label,
                            "config": config_dir,
                            "seed": seed,
                            "fitness_gen0": fitness_gen0,
                            "fitness_final": final_fitness,
                            "avg_fitness_final": final_avg,
                            "fitness_std_final": final_std,
                            "gen_total": total_gens,
                            "gen_90pct": gen_90,
                            "gen_95pct": gen_95,
                            "elapsed_seconds": elapsed,
                        }
                    )
                except Exception as e:
                    print(f"  Error leyendo {csv_path}: {e}")

    return pd.DataFrame(rows)


def plot_fitness_heatmap(
    data: pd.DataFrame,
    title: str,
    output_path: str,
    y_label: str = "Configuracion",
):
    """Heatmap: config x imagen -> fitness final medio."""
    pivot = data.groupby(["config", "imagen"])["fitness_final"].mean().unstack()
    pivot = pivot.loc[sorted(pivot.index, key=_natural_sort_key)]
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=_natural_sort_key))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(4.8, len(pivot) * 1.05)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="RdYlGn",
        linewidths=0.5,
        vmin=pivot.values.min() * 0.98,
        vmax=min(1.0, pivot.values.max() * 1.005),
        cbar_kws={"label": "Fitness final"},
        ax=ax,
    )
    ax.set_title(f"Fitness final por configuracion e imagen\n{title}", fontsize=13)
    ax.set_xlabel("Imagen", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11, labelpad=24)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_yticklabels(
        [_pretty_config_label(t.get_text()) for t in ax.get_yticklabels()],
        rotation=0,
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap guardado: {output_path}")


def plot_convergence_heatmap(
    data: pd.DataFrame,
    title: str,
    output_path: str,
    y_label: str = "Configuracion",
):
    """Heatmap: config x imagen -> generacion para alcanzar 90% fitness."""
    pivot = data.groupby(["config", "imagen"])["gen_90pct"].mean().unstack()
    pivot = pivot.loc[sorted(pivot.index, key=_natural_sort_key)]
    pivot = pivot.reindex(columns=sorted(pivot.columns, key=_natural_sort_key))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2.5), max(4.8, len(pivot) * 1.05)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        linewidths=0.5,
        cbar_kws={"label": "Generaciones para 90%"},
        ax=ax,
    )
    ax.set_title(f"Velocidad de convergencia (gen para 90%)\n{title}", fontsize=13)
    ax.set_xlabel("Imagen", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11, labelpad=24)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.set_yticklabels(
        [_pretty_config_label(t.get_text()) for t in ax.get_yticklabels()],
        rotation=0,
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.22)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap convergencia guardado: {output_path}")


def _images_label(n_images: int) -> str:
    return "imagen" if n_images == 1 else "imagenes"


def plot_grouped_bars(data: pd.DataFrame, title: str, output_path: str):
    """Barras agrupadas: fitness final por config, una barra por imagen."""
    summary = data.groupby(["config", "imagen"])["fitness_final"].agg(["mean", "std"]).reset_index()

    configs = sorted(data["config"].unique(), key=_natural_sort_key)
    images = sorted(data["imagen"].unique(), key=_natural_sort_key)
    n_configs = len(configs)
    n_images = len(images)

    x = np.arange(n_configs)
    width = 0.8 / n_images
    colors = plt.cm.Set2(np.linspace(0, 1, n_images))

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 6))

    for i, img in enumerate(images):
        img_data = summary[summary["imagen"] == img].set_index("config")
        means = [img_data.loc[c, "mean"] if c in img_data.index else 0 for c in configs]
        stds = [img_data.loc[c, "std"] if c in img_data.index else 0 for c in configs]
        ax.bar(
            x + i * width,
            means,
            width,
            label=img,
            color=colors[i],
            yerr=stds,
            capsize=4,
            error_kw={"linewidth": 1.5, "capthick": 1.5},
        )

    ax.set_xlabel("Configuracion", fontsize=11)
    ax.set_ylabel("Fitness final", fontsize=11)
    ax.set_title(
        f"Comparacion de fitness final en {n_images} {_images_label(n_images)}\n{title}",
        fontsize=13,
    )
    ax.set_xticks(x + width * (n_images - 1) / 2)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=9, title="Imagen")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Barras agrupadas guardado: {output_path}")


def plot_gen0_bars(data: pd.DataFrame, title: str, output_path: str):
    """Barras agrupadas: fitness en generacion 0 por config, una barra por imagen."""
    if "fitness_gen0" not in data.columns:
        return

    summary = data.groupby(["config", "imagen"])["fitness_gen0"].agg(["mean", "std"]).reset_index()

    configs = sorted(data["config"].unique(), key=_natural_sort_key)
    images = sorted(data["imagen"].unique(), key=_natural_sort_key)
    n_configs = len(configs)
    n_images = len(images)

    x = np.arange(n_configs)
    width = 0.8 / n_images
    colors = plt.cm.Set2(np.linspace(0, 1, n_images))

    fig, ax = plt.subplots(figsize=(max(10, n_configs * 1.5), 6))

    for i, img in enumerate(images):
        img_data = summary[summary["imagen"] == img].set_index("config")
        means = [img_data.loc[c, "mean"] if c in img_data.index else 0 for c in configs]
        stds = [img_data.loc[c, "std"] if c in img_data.index else 0 for c in configs]
        ax.bar(
            x + i * width,
            means,
            width,
            label=img,
            color=colors[i],
            yerr=stds,
            capsize=4,
            error_kw={"linewidth": 1.5, "capthick": 1.5},
        )

    ax.set_xlabel("Configuracion", fontsize=11)
    ax.set_ylabel("Fitness en generacion 0", fontsize=11)
    ax.set_title(
        f"Fitness inicial (gen 0) en {n_images} {_images_label(n_images)}\n{title}",
        fontsize=13,
    )
    ax.set_xticks(x + width * (n_images - 1) / 2)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=9, title="Imagen")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Barras gen0 guardado: {output_path}")


def save_cross_summary(data: pd.DataFrame, output_path: str):
    """Tabla resumen CSV con fitness y velocidad por config x imagen."""
    summary = data.groupby(["config", "imagen"]).agg(
        fitness_mean=("fitness_final", "mean"),
        fitness_std=("fitness_final", "std"),
        gen_90_mean=("gen_90pct", "mean"),
        gen_95_mean=("gen_95pct", "mean"),
        time_mean=("elapsed_seconds", "mean"),
    ).round(4).reset_index()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(f"  Tabla resumen guardada: {output_path}")


def process_experiment(
    results_dir: str,
    output_dir: str,
    exp_name: str,
    include_configs: list[str] | None = None,
    include_images: list[str] | None = None,
    image_renames: dict[str, str] | None = None,
):
    """Procesa un experimento con estructura por imagen."""
    print(f"\n  Procesando analisis cruzado: {exp_name}")

    include_images_set = set(include_images) if include_images else None

    data = load_cross_image_data(
        results_dir,
        include_images=include_images_set,
        image_renames=image_renames,
    )

    if include_configs:
        include_set = set(include_configs)
        data = data[data["config"].isin(include_set)]

    if data.empty:
        print(f"  Sin datos en {results_dir}")
        return

    y_axis_label = "Cantidad de triangulos" if exp_name == "num_triangulos" else "Configuracion"

    plot_fitness_heatmap(
        data,
        exp_name,
        os.path.join(output_dir, f"{exp_name}_heatmap_fitness.png"),
        y_label=y_axis_label,
    )

    plot_convergence_heatmap(
        data,
        exp_name,
        os.path.join(output_dir, f"{exp_name}_heatmap_convergencia.png"),
        y_label=y_axis_label,
    )

    plot_grouped_bars(
        data,
        exp_name,
        os.path.join(output_dir, f"{exp_name}_barras_comparacion.png"),
    )

    plot_gen0_bars(
        data,
        exp_name,
        os.path.join(output_dir, f"{exp_name}_barras_gen0.png"),
    )

    save_cross_summary(
        data,
        os.path.join(output_dir, f"{exp_name}_cross_summary.csv"),
    )


def main():
    parser = argparse.ArgumentParser(description="Analisis cruzado entre imagenes")
    parser.add_argument("--input", required=True, help="Directorio de resultados")
    parser.add_argument("--output", required=True, help="Directorio de plots")
    parser.add_argument("--all", action="store_true", help="Procesar todos los subdirectorios")
    parser.add_argument("--name", default=None, help="Nombre del experimento")
    parser.add_argument(
        "--include_configs",
        default="",
        help="Lista separada por comas de configuraciones a incluir (ej: cfg1,cfg2)",
    )
    parser.add_argument(
        "--include_images",
        default="",
        help="Lista separada por comas de imagenes a incluir (ej: 1_Ucrania,3_Apple)",
    )
    parser.add_argument(
        "--image_renames",
        default="",
        help="Mapping imagen_origen:imagen_alias separado por comas",
    )

    args = parser.parse_args()
    include_configs = _parse_csv_list(args.include_configs)
    include_images = _parse_csv_list(args.include_images)
    image_renames = _parse_image_renames(args.image_renames)

    if args.all:
        for exp_name in sorted(os.listdir(args.input), key=_natural_sort_key):
            exp_path = os.path.join(args.input, exp_name)
            if os.path.isdir(exp_path) and exp_name != "complejidad":
                process_experiment(
                    exp_path,
                    os.path.join(args.output, exp_name),
                    exp_name,
                    include_configs=include_configs,
                    include_images=include_images,
                    image_renames=image_renames,
                )
    else:
        exp_name = args.name or os.path.basename(args.input.rstrip("/"))
        process_experiment(
            args.input,
            args.output,
            exp_name,
            include_configs=include_configs,
            include_images=include_images,
            image_renames=image_renames,
        )


if __name__ == "__main__":
    main()
