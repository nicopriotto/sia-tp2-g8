"""
Genera graficos comparativos a partir de los resultados de los experimentos.

Salida:
- Curvas fitness vs generacion (una linea por configuracion, banda de error con std)
- Grid de imagenes: gen_0000, gen_0050, gen_0200, final (una fila por configuracion)
- Tabla comparativa: config | fitness_final_mean | fitness_final_std | generaciones para 90%

Uso:
    python experiments/plot_results.py --input experiments/results/seleccion/ --output experiments/plots/seleccion/
    python experiments/plot_results.py --input experiments/results/ --all
"""
import argparse
import os
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


def load_metrics(results_dir: str) -> dict[str, list]:
    """
    Carga todos los metrics.csv de un directorio de resultados.

    Estructura esperada:
        results_dir/
            config_1/
                seed_42/metrics.csv
                seed_123/metrics.csv
            config_2/
                seed_42/metrics.csv

    Retorna: {nombre_config: [DataFrame, ...]}
    """
    data = {}
    if not os.path.isdir(results_dir):
        print(f"Directorio no encontrado: {results_dir}")
        return data

    for config_dir in sorted(os.listdir(results_dir)):
        config_path = os.path.join(results_dir, config_dir)
        if not os.path.isdir(config_path):
            continue

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

        plt.plot(mean.index, mean.values, label=label, linewidth=1.5)
        plt.fill_between(
            mean.index,
            (mean.values - std.values).clip(0),
            (mean.values + std.values).clip(0, 1),
            alpha=0.15,
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


def plot_avg_fitness_curves(data: dict, title: str, output_path: str):
    """Grafica curvas de fitness promedio de la poblacion."""
    if not data:
        print("Sin datos para graficar curvas de fitness promedio.")
        return

    plt.figure(figsize=(12, 6))

    for label, dfs in data.items():
        merged = pd.concat(dfs).groupby("generation")["avg_fitness"]
        mean = merged.mean()
        std = merged.std().fillna(0)

        plt.plot(mean.index, mean.values, label=label, linewidth=1.5)
        plt.fill_between(
            mean.index,
            (mean.values - std.values).clip(0),
            (mean.values + std.values).clip(0, 1),
            alpha=0.15,
        )

    plt.xlabel("Generacion", fontsize=12)
    plt.ylabel("Fitness promedio de la poblacion", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Grafico guardado: {output_path}")


def create_image_grid(results_dir: str, generations: list[int], output_path: str):
    """
    Crea un grid de imagenes intermedias.
    Una fila por configuracion, una columna por generacion.
    Usa la primera seed disponible para cada configuracion.
    """
    if not os.path.isdir(results_dir):
        print(f"Directorio no encontrado: {results_dir}")
        return

    configs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ])

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

        # Usar la primera seed
        seed_path = os.path.join(config_path, seed_dirs[0])
        row_images = []

        for gen in generations:
            img_path = os.path.join(seed_path, f"gen_{gen:04d}.png")
            if not os.path.exists(img_path):
                # Intentar con final.png si es la ultima generacion
                img_path = os.path.join(seed_path, "final.png")

            if os.path.exists(img_path):
                row_images.append(Image.open(img_path))
            else:
                row_images.append(None)

        rows.append((config_name, row_images))

    if not rows:
        print("Sin imagenes para crear grid.")
        return

    # Determinar tamano de celda
    cell_w, cell_h = 100, 100
    for _, images in rows:
        for img in images:
            if img is not None:
                cell_w, cell_h = img.size
                break

    n_cols = len(generations)
    n_rows = len(rows)
    label_width = 200  # espacio para nombre de la configuracion
    padding = 5

    grid_w = label_width + n_cols * (cell_w + padding)
    grid_h = 30 + n_rows * (cell_h + padding)  # 30px para header

    grid = Image.new("RGB", (grid_w, grid_h), "white")

    # Componer el grid
    for row_idx, (config_name, images) in enumerate(rows):
        y = 30 + row_idx * (cell_h + padding)
        for col_idx, img in enumerate(images):
            x = label_width + col_idx * (cell_w + padding)
            if img is not None:
                img_resized = img.resize((cell_w, cell_h))
                grid.paste(img_resized, (x, y))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grid.save(output_path)
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


def process_experiment(results_dir: str, output_dir: str, experiment_name: str):
    """Procesa un experimento completo: carga datos, genera graficos y tabla."""
    print(f"\n{'='*60}")
    print(f"Procesando experimento: {experiment_name}")
    print(f"{'='*60}")

    data = load_metrics(results_dir)
    if not data:
        print(f"Sin resultados en {results_dir}")
        return

    # Curvas de fitness del mejor individuo
    plot_fitness_curves(
        data,
        title=f"Fitness del mejor individuo - {experiment_name}",
        output_path=os.path.join(output_dir, "fitness_best.png"),
    )

    # Curvas de fitness promedio
    plot_avg_fitness_curves(
        data,
        title=f"Fitness promedio - {experiment_name}",
        output_path=os.path.join(output_dir, "fitness_avg.png"),
    )

    # Grid de imagenes intermedias
    create_image_grid(
        results_dir,
        generations=[0, 50, 200, 500],
        output_path=os.path.join(output_dir, "image_grid.png"),
    )

    # Tabla resumen
    create_summary_table(
        data,
        output_path=os.path.join(output_dir, "summary.csv"),
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

    args = parser.parse_args()

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
                )
    else:
        exp_name = args.name or os.path.basename(args.input.rstrip("/"))
        output_dir = args.output or os.path.join(
            os.path.dirname(args.input.rstrip("/")), "..", "plots", exp_name
        )
        process_experiment(args.input, output_dir, exp_name)


if __name__ == "__main__":
    main()
