"""
Genera graficos comparativos a partir de los resultados de los experimentos.

Salida:
- Curvas fitness vs generacion (una linea por configuracion, banda de error con std)
- Grid de imagenes: gen_0000, gen_0050, gen_0200, gen_0500, gen_3000/final (una fila por configuracion)
- Tabla comparativa: config | fitness_final_mean | fitness_final_std | generaciones para 90%

Uso:
    python experiments/plot_results.py --input experiments/results/seleccion/ --output experiments/plots/seleccion/
    python experiments/plot_results.py --input experiments/results/ --all
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
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: dependencia no instalada: {e}")
    print("Instalar con: pip3 install matplotlib pandas Pillow")
    sys.exit(1)


def _natural_sort_key(name: str):
    """Ordena configs de forma logica: numeros por valor, texto alfabetico."""
    parts = re.split(r'(\d+)', name)
    result = []
    for part in parts:
        if part.isdigit():
            result.append(int(part))
        else:
            result.append(part.lower())
    return result


def load_metrics(results_dir: str, include_configs: set[str] | None = None) -> dict[str, list]:
    """
    Carga todos los metrics.csv de un directorio de resultados.

    Estructura esperada:
        results_dir/
            config_1/
                seed_42/metrics.csv
                seed_123/metrics.csv
            config_2/
                seed_42/metrics.csv

    Retorna: {nombre_config: [DataFrame, ...]} ordenado logicamente.
    """
    data = {}
    if not os.path.isdir(results_dir):
        print(f"Directorio no encontrado: {results_dir}")
        return data

    config_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if include_configs:
        config_dirs = [d for d in config_dirs if d in include_configs]
    config_dirs = sorted(config_dirs, key=_natural_sort_key)

    for config_dir in config_dirs:
        config_path = os.path.join(results_dir, config_dir)

        dfs = []
        for seed_dir in sorted(os.listdir(config_path)):
            seed_path = os.path.join(config_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            csv_path = os.path.join(seed_path, "metrics.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        dfs.append(df)
                except Exception as e:
                    print(f"  Error leyendo {csv_path}: {e}")

        if dfs:
            data[config_dir] = dfs

    return data


def plot_fitness_curves(data: dict, title: str, output_path: str):
    """Grafica curvas de fitness promedio con banda de desviacion estandar."""
    if not data:
        print("Sin datos para graficar curvas de fitness.")
        return

    plt.figure(figsize=(12, 6))

    for label, dfs in data.items():
        # Alinear por generacion y promediar entre seeds
        merged = pd.concat(dfs).groupby("generation")["best_fitness"]
        mean = merged.mean()
        std = merged.std().fillna(0)

        line, = plt.plot(mean.index, mean.values, label=label, linewidth=1.5)
        plt.fill_between(
            mean.index,
            (mean.values - std.values).clip(0),
            (mean.values + std.values).clip(0, 1),
            alpha=0.08, color=line.get_color(),
        )

    plt.xlabel("Generacion", fontsize=12)
    plt.ylabel("Fitness (mejor individuo)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico guardado: {output_path}")


def plot_avg_fitness_small_multiples(
    data: dict,
    title: str,
    output_path: str,
    avg_mode: str = "raw",
    n_cols: int = 3,
):
    """Grafica fitness_avg en small multiples: un subplot por configuracion."""
    if not data:
        print("Sin datos para graficar small multiples de fitness promedio.")
        return

    if avg_mode not in {"raw", "fitness_log", "error_log"}:
        raise ValueError(
            f"avg_mode invalido: {avg_mode}. Opciones: raw, fitness_log, error_log"
        )

    labels = list(data.keys())
    n_plots = len(labels)
    n_cols = max(1, min(n_cols, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))
    min_positive = 1e-8

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes, dtype=object)
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        merged_df = pd.concat(data[label])
        if avg_mode == "fitness_log":
            merged_df["plot_value"] = merged_df["avg_fitness"].clip(lower=min_positive)
        elif avg_mode == "error_log":
            merged_df["plot_value"] = (1.0 - merged_df["avg_fitness"]).clip(lower=min_positive)
        else:
            merged_df["plot_value"] = merged_df["avg_fitness"]

        merged = merged_df.groupby("generation")["plot_value"]
        mean = merged.mean()
        std = merged.std().fillna(0)

        ax.plot(mean.index, mean.values, linewidth=1.6)
        if avg_mode in {"fitness_log", "error_log"}:
            lower = np.maximum(mean.values - std.values, min_positive)
            upper = np.maximum(mean.values + std.values, lower)
        else:
            lower = (mean.values - std.values).clip(0)
            upper = (mean.values + std.values).clip(0, 1)
        ax.fill_between(mean.index, lower, upper, alpha=0.08)

        if avg_mode in {"fitness_log", "error_log"}:
            ax.set_yscale("log")

        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3, which="both")

        if row == n_rows - 1:
            ax.set_xlabel("Generacion", fontsize=9)

        if col == 0:
            if avg_mode == "fitness_log":
                ax.set_ylabel("Fitness [log]", fontsize=9)
            elif avg_mode == "error_log":
                ax.set_ylabel("Error (1 - fitness) [log]", fontsize=9)
            else:
                ax.set_ylabel("Fitness promedio", fontsize=9)

    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis("off")

    fig.suptitle(f"{title} (small multiples)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Grafico small multiples guardado: {output_path}")


def plot_avg_fitness_curves(
    data: dict,
    title: str,
    output_path: str,
    avg_mode: str = "raw",
):
    """Grafica curvas de fitness promedio con modos raw, log(fitness) o log(1-fitness)."""
    if not data:
        print("Sin datos para graficar curvas de fitness promedio.")
        return

    if avg_mode not in {"raw", "fitness_log", "error_log"}:
        raise ValueError(
            f"avg_mode invalido: {avg_mode}. Opciones: raw, fitness_log, error_log"
        )

    plt.figure(figsize=(12, 6))
    min_positive = 1e-8

    for label, dfs in data.items():
        merged_df = pd.concat(dfs)

        if avg_mode == "fitness_log":
            merged_df["plot_value"] = merged_df["avg_fitness"].clip(lower=min_positive)
        elif avg_mode == "error_log":
            # Separar curvas cerca de 1.0: trabajar en error por seed y luego promediar.
            merged_df["plot_value"] = (1.0 - merged_df["avg_fitness"]).clip(lower=min_positive)
        else:
            merged_df["plot_value"] = merged_df["avg_fitness"]

        merged = merged_df.groupby("generation")["plot_value"]
        mean = merged.mean()
        std = merged.std().fillna(0)

        line, = plt.plot(mean.index, mean.values, label=label, linewidth=1.5)

        if avg_mode in {"fitness_log", "error_log"}:
            lower = np.maximum(mean.values - std.values, min_positive)
            upper = np.maximum(mean.values + std.values, lower)
        else:
            lower = (mean.values - std.values).clip(0)
            upper = (mean.values + std.values).clip(0, 1)
        plt.fill_between(
            mean.index,
            lower,
            upper,
            alpha=0.08, color=line.get_color(),
        )

    plt.xlabel("Generacion", fontsize=12)
    if avg_mode == "fitness_log":
        plt.yscale("log")
        plt.ylabel("Fitness promedio [log]", fontsize=12)
    elif avg_mode == "error_log":
        plt.yscale("log")
        plt.ylabel("Error promedio (1 - fitness) [log]", fontsize=12)
    else:
        plt.ylabel("Fitness promedio de la poblacion", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico guardado: {output_path}")




def plot_avg_error_gap_to_best(data: dict, title: str, output_path: str):
    """Grafica la brecha de error (1-fitness) de cada config respecto al mejor metodo."""
    if not data:
        print("Sin datos para graficar brecha de error respecto al mejor.")
        return

    avg_curves = {}
    final_best_score = {}

    for label, dfs in data.items():
        merged = pd.concat(dfs)
        avg_curve = merged.groupby("generation")["avg_fitness"].mean()
        best_curve = merged.groupby("generation")["best_fitness"].mean()
        avg_curves[label] = avg_curve
        final_best_score[label] = best_curve.iloc[-1]

    best_label = max(final_best_score, key=final_best_score.get)
    base_error = 1.0 - avg_curves[best_label]

    plt.figure(figsize=(12, 6))

    labels = sorted(avg_curves.keys(), key=_natural_sort_key)
    for label in labels:
        curve = avg_curves[label]
        idx = curve.index.intersection(base_error.index)
        gap = (1.0 - curve.loc[idx]) - base_error.loc[idx]
        plt.plot(
            idx,
            gap.values,
            label=label,
            linewidth=2.0 if label == best_label else 1.5,
            alpha=1.0 if label == best_label else 0.9,
        )

    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Generacion", fontsize=12)
    plt.ylabel(f"Brecha de error vs {best_label} (1 - fitness)", fontsize=12)
    plt.title(f"Diferencia de error respecto al mejor metodo\n{title}", fontsize=14)
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico gap-to-best guardado: {output_path}")

def create_image_grid(
    results_dir: str,
    generations: list[int],
    output_path: str,
    include_configs: set[str] | None = None,
):
    """
    Crea un grid de imagenes intermedias con matplotlib.
    Una fila por configuracion (ordenada logicamente), una columna por generacion.
    Labels claros en filas (nombre config) y columnas (generacion).
    Usa la primera seed disponible para cada configuracion.
    """
    if not os.path.isdir(results_dir):
        print(f"Directorio no encontrado: {results_dir}")
        return

    configs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if include_configs:
        configs = [d for d in configs if d in include_configs]
    configs = sorted(configs, key=_natural_sort_key)

    if not configs:
        print("Sin configuraciones para crear grid de imagenes.")
        return

    # Recolectar imagenes disponibles
    rows = []
    for config_name in configs:
        config_path = os.path.join(results_dir, config_name)
        seed_dirs = sorted([
            d for d in os.listdir(config_path)
            if os.path.isdir(os.path.join(config_path, d))
        ])

        if not seed_dirs:
            continue

        seed_path = os.path.join(config_path, seed_dirs[0])
        row_images = []

        for gen in generations:
            img_path = os.path.join(seed_path, f"gen_{gen:04d}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(seed_path, "final.png")

            if os.path.exists(img_path):
                row_images.append(np.array(Image.open(img_path)))
            else:
                row_images.append(None)

        rows.append((config_name, row_images))

    if not rows:
        print("Sin imagenes para crear grid.")
        return

    n_rows = len(rows)
    n_cols = len(generations)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

    # Asegurar que axes sea 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, (config_name, images) in enumerate(rows):
        for col_idx, img_arr in enumerate(images):
            ax = axes[row_idx, col_idx]
            if img_arr is not None:
                ax.imshow(img_arr)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

            # Label de columna (generacion) en la primera fila
            if row_idx == 0:
                ax.set_title(f"Gen {generations[col_idx]}", fontsize=11, fontweight="bold")

        # Label de fila (config) a la izquierda
        axes[row_idx, 0].annotate(
            config_name, xy=(-0.1, 0.5), xycoords="axes fraction",
            fontsize=10, ha="right", va="center", fontweight="bold",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grid de imagenes guardado: {output_path}")


def create_summary_table(data: dict, output_path: str):
    """
    Genera tabla resumen con fitness final medio, std, y generaciones
    para alcanzar el 90% del fitness final.
    """
    if not data:
        print("Sin datos para generar tabla resumen.")
        return

    rows = []
    for label, dfs in data.items():
        finals = [df["best_fitness"].iloc[-1] for df in dfs]
        total_gens = [len(df) - 1 for df in dfs]  # -1 porque gen 0 es la inicial

        # Calcular generaciones para alcanzar 90% del fitness final
        gens_90 = []
        for df in dfs:
            target_90 = df["best_fitness"].iloc[-1] * 0.9
            reached = df[df["best_fitness"] >= target_90]
            if len(reached) > 0:
                gens_90.append(reached["generation"].iloc[0])

        rows.append({
            "configuracion": label,
            "fitness_final_mean": round(np.mean(finals), 6),
            "fitness_final_std": round(np.std(finals), 6),
            "generaciones_mean": round(np.mean(total_gens), 1),
            "gen_90pct_mean": round(np.mean(gens_90), 1) if gens_90 else "N/A",
        })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("TABLA RESUMEN")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    print(f"\nTabla guardada: {output_path}")


def process_experiment(
    results_dir: str,
    output_dir: str,
    experiment_name: str,
    avg_mode: str = "raw",
    include_configs: list[str] | None = None,
    avg_small_multiples: bool = False,
    avg_gap_to_best: bool = False,
):
    """Procesa un experimento completo: carga datos, genera graficos y tabla."""
    print(f"\n{'='*60}")
    print(f"Procesando experimento: {experiment_name}")
    print(f"{'='*60}")

    include_set = set(include_configs) if include_configs else None

    data = load_metrics(results_dir, include_configs=include_set)
    if not data:
        print(f"Sin resultados en {results_dir}")
        return

    # Nombre limpio para archivos: "seleccion - 2_Grecia" -> "seleccion_2_Grecia"
    safe_name = experiment_name.replace(" - ", "_").replace(" ", "_")

    # Curvas de fitness del mejor individuo
    plot_fitness_curves(
        data,
        title=f"Fitness del mejor individuo - {experiment_name}",
        output_path=os.path.join(output_dir, f"{safe_name}_fitness_best.png"),
    )

    # Curvas de fitness promedio
    plot_avg_fitness_curves(
        data,
        title=f"Fitness promedio - {experiment_name}",
        output_path=os.path.join(output_dir, f"{safe_name}_fitness_avg.png"),
        avg_mode=avg_mode,
    )
    if avg_small_multiples:
        plot_avg_fitness_small_multiples(
            data,
            title=f"Fitness promedio - {experiment_name}",
            output_path=os.path.join(output_dir, f"{safe_name}_fitness_avg_small_multiples.png"),
            avg_mode=avg_mode,
        )
    if avg_gap_to_best:
        plot_avg_error_gap_to_best(
            data,
            title=f"Fitness promedio - {experiment_name}",
            output_path=os.path.join(output_dir, f"{safe_name}_fitness_avg_gap_best.png"),
        )

    # Grid de imagenes intermedias
    create_image_grid(
        results_dir,
        generations=[0, 50, 200, 500, 3000],
        output_path=os.path.join(output_dir, f"{safe_name}_image_grid.png"),
        include_configs=include_set,
    )

    # Tabla resumen
    create_summary_table(
        data,
        output_path=os.path.join(output_dir, f"{safe_name}_summary.csv"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Genera graficos comparativos a partir de resultados de experimentos"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Directorio con resultados del experimento"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directorio de salida para graficos (default: plots/ junto a input)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Nombre del experimento (default: nombre del directorio de input)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Procesar todos los subdirectorios como experimentos independientes"
    )
    parser.add_argument(
        "--avg_mode", type=str, choices=["raw", "fitness_log", "error_log"], default="raw",
        help="Modo del grafico fitness_avg: raw | fitness_log | error_log"
    )
    parser.add_argument(
        "--include_configs", type=str, default="",
        help="Lista separada por comas de configuraciones a incluir (ej: cfg1,cfg2)"
    )
    parser.add_argument(
        "--avg_small_multiples", action="store_true",
        help="Genera un grafico adicional de fitness_avg con subplots (uno por configuracion)"
    )
    parser.add_argument(
        "--avg_gap_to_best", action="store_true",
        help="Genera un grafico adicional con la brecha de error de cada config respecto al mejor metodo"
    )

    args = parser.parse_args()
    include_configs = [c.strip() for c in args.include_configs.split(",") if c.strip()]

    if args.all:
        # Procesar todos los subdirectorios
        base_dir = args.input
        output_base = args.output or os.path.join(os.path.dirname(base_dir.rstrip("/")), "plots")

        for exp_name in sorted(os.listdir(base_dir)):
            exp_path = os.path.join(base_dir, exp_name)
            if os.path.isdir(exp_path):
                process_experiment(
                    exp_path,
                    os.path.join(output_base, exp_name),
                    exp_name,
                    avg_mode=args.avg_mode,
                    include_configs=include_configs,
                    avg_small_multiples=args.avg_small_multiples,
                    avg_gap_to_best=args.avg_gap_to_best,
                )
    else:
        exp_name = args.name or os.path.basename(args.input.rstrip("/"))
        output_dir = args.output or os.path.join(
            os.path.dirname(args.input.rstrip("/")), "..", "plots", exp_name
        )
        process_experiment(
            args.input,
            output_dir,
            exp_name,
            avg_mode=args.avg_mode,
            include_configs=include_configs,
            avg_small_multiples=args.avg_small_multiples,
            avg_gap_to_best=args.avg_gap_to_best,
        )


if __name__ == "__main__":
    main()
