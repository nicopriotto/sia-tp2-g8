"""
Genera graficos para los resultados del overnight run de islas + noche estrellada.

Lee los metrics.csv de output/ y genera:
- Curvas de fitness de las 6 islas superpuestas
- Grid de snapshots (gen 0, 500, 1000, 2000, 3000)
- Imagen final
"""
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
    sys.exit(1)


def find_output_dir():
    """Busca el directorio de output con resultados de islas."""
    for d in ["output", "output_backup_run2"]:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.isdir(path):
            # Verificar que tenga subdirectorios de islas
            subdirs = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
            if len(subdirs) >= 4:
                return path
    return None


def main():
    output_dir = find_output_dir()
    if output_dir is None:
        print("No se encontraron resultados de islas en output/ ni output_backup_run2/")
        return

    plot_dir = os.path.join(PROJECT_ROOT, "experiments", "plots", "islas_noche")
    os.makedirs(plot_dir, exist_ok=True)

    # --- Curvas de fitness ---
    plt.figure(figsize=(14, 6))

    island_dirs = sorted([
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ])

    for island_name in island_dirs:
        csv_path = os.path.join(output_dir, island_name, "metrics.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 2:
            continue

        plt.plot(df["generation"], df["best_fitness"], label=island_name, linewidth=1.2)

    plt.xlabel("Generacion", fontsize=12)
    plt.ylabel("Fitness (mejor individuo)", fontsize=12)
    plt.title("Modelo de Islas - Noche Estrellada", fontsize=14)
    plt.legend(fontsize=9, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fitness_path = os.path.join(plot_dir, "islas_noche_fitness.png")
    plt.savefig(fitness_path, dpi=150)
    plt.close()
    print(f"Grafico guardado: {fitness_path}")

    # --- Grid de snapshots ---
    target_gens = [0, 500, 1000, 2000, 3000]
    # Usar la isla con mejor fitness final
    best_island = None
    best_fitness = 0

    for island_name in island_dirs:
        csv_path = os.path.join(output_dir, island_name, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if len(df) > 0 and df["best_fitness"].iloc[-1] > best_fitness:
            best_fitness = df["best_fitness"].iloc[-1]
            best_island = island_name

    if best_island:
        island_path = os.path.join(output_dir, best_island)
        images = []
        labels = []

        for gen in target_gens:
            img_path = os.path.join(island_path, f"gen_{gen:04d}.png")
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
                labels.append(f"Gen {gen}")

        # Agregar imagen original
        original_path = os.path.join(PROJECT_ROOT, "images", "noche.jpg")
        if not os.path.exists(original_path):
            original_path = os.path.join(PROJECT_ROOT, "images", "5.jpg")

        if images:
            n = len(images)
            fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

            if os.path.exists(original_path):
                orig = Image.open(original_path)
                axes[0].imshow(orig)
                axes[0].set_title("Original", fontsize=11)
                axes[0].axis("off")

            for i, (img, label) in enumerate(zip(images, labels)):
                axes[i + 1].imshow(img)
                axes[i + 1].set_title(label, fontsize=11)
                axes[i + 1].axis("off")

            plt.suptitle(f"Mejor isla: {best_island} (fitness={best_fitness:.4f})", fontsize=13)
            plt.tight_layout()

            grid_path = os.path.join(plot_dir, "islas_noche_snapshots.png")
            plt.savefig(grid_path, dpi=150)
            plt.close()
            print(f"Grid guardado: {grid_path}")

    # --- Diversidad (std) por isla ---
    plt.figure(figsize=(14, 6))

    for island_name in island_dirs:
        csv_path = os.path.join(output_dir, island_name, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        if len(df) < 2:
            continue
        plt.plot(df["generation"], df["fitness_std"], label=island_name, linewidth=1.0)

    plt.xlabel("Generacion", fontsize=12)
    plt.ylabel("Desviacion estandar del fitness", fontsize=12)
    plt.title("Diversidad por isla - Noche Estrellada", fontsize=14)
    plt.legend(fontsize=9, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    std_path = os.path.join(plot_dir, "islas_noche_diversidad.png")
    plt.savefig(std_path, dpi=150)
    plt.close()
    print(f"Grafico guardado: {std_path}")

    print(f"\nTodos los plots en: {plot_dir}")


if __name__ == "__main__":
    main()
