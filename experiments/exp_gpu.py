"""
Experimento: CPU vs GPU.

Mide el tiempo por generacion con CPU y GPU para distintas cantidades
de triangulos (50, 100, 200, 400) en las 4 imagenes principales.
"""
import os
import sys
import time
import logging
import json

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.run_experiment import BASE_CONFIG, ALL_IMAGES, load_target_image

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "experiments", "results", "gpu")

TRIANGLE_COUNTS = [50, 100, 200, 400]
N_GENERATIONS = 200


def benchmark_renderer(config_dict, target, output_dir):
    """Corre N generaciones y mide tiempo promedio por generacion."""
    import random as rand
    from config.config_loader import Config
    from core.genetic_algorithm import GeneticAlgorithm
    from core.ga_context import GAContext
    from main import build_operators, create_renderer

    rand.seed(42)
    np.random.seed(42)

    config = Config(**config_dict)
    height, width = target.shape[0], target.shape[1]

    renderer = create_renderer(config, target, width, height)
    selection_ops, crossover_ops, mutation_ops, survival, fitness_fn = build_operators(config)
    context = GAContext(generation=0, max_generations=config.max_generations)

    for sel in selection_ops:
        if hasattr(sel, "context"):
            sel.context = context
    for mut in mutation_ops:
        if hasattr(mut, "context"):
            mut.context = context

    ga = GeneticAlgorithm(
        config=config, target_image=target, renderer=renderer,
        fitness_fn=fitness_fn, selection_ops=selection_ops,
        crossover_ops=crossover_ops, mutation_ops=mutation_ops,
        survival=survival, context=context,
        output_dir=output_dir,
    )

    t0 = time.time()
    result = ga.run()
    elapsed = time.time() - t0

    avg_time = elapsed / max(result.final_generation, 1)
    return avg_time, result.best_individual.fitness


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    os.makedirs(DEFAULT_OUTPUT, exist_ok=True)

    all_results = {}

    for img_label, image_path in ALL_IMAGES:
        target = load_target_image(image_path)
        results = []

        for n_tri in TRIANGLE_COUNTS:
            for use_gpu in [False, True]:
                renderer_label = "GPU" if use_gpu else "CPU"
                label = f"{renderer_label}_{n_tri}tri"
                config_dict = {
                    **BASE_CONFIG,
                    "triangle_count": n_tri,
                    "max_generations": N_GENERATIONS,
                    "use_gpu": use_gpu,
                    "gpu_device": "dedicated" if use_gpu else "auto",
                    "save_every": 9999,
                }

                tmp_dir = os.path.join(DEFAULT_OUTPUT, img_label, "_tmp")
                print(f"[gpu] {img_label}/{label}...")
                try:
                    avg_time, fitness = benchmark_renderer(config_dict, target, tmp_dir)
                    results.append({
                        "triangle_count": n_tri,
                        "renderer": renderer_label,
                        "avg_time_per_gen": round(avg_time, 4),
                        "final_fitness": round(fitness, 6),
                    })
                    print(f"  {avg_time:.4f}s/gen, fitness={fitness:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        all_results[img_label] = results

    # Guardar resultados
    output_path = os.path.join(DEFAULT_OUTPUT, "benchmark.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generar plots por imagen
    try:
        import matplotlib.pyplot as plt

        for img_label, results in all_results.items():
            if not results:
                continue

            cpu_times = {r["triangle_count"]: r["avg_time_per_gen"] for r in results if r["renderer"] == "CPU"}
            gpu_times = {r["triangle_count"]: r["avg_time_per_gen"] for r in results if r["renderer"] == "GPU"}

            tri_counts = sorted(cpu_times.keys())
            x = np.arange(len(tri_counts))
            width = 0.35

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            cpu_vals = [cpu_times[t] for t in tri_counts]
            gpu_vals = [gpu_times.get(t, 0) for t in tri_counts]

            ax1.bar(x - width/2, cpu_vals, width, label="CPU", color="#e74c3c")
            ax1.bar(x + width/2, gpu_vals, width, label="GPU", color="#2ecc71")
            ax1.set_xlabel("Triangulos")
            ax1.set_ylabel("Tiempo por generacion (s)")
            ax1.set_title(f"CPU vs GPU - {img_label}")
            ax1.set_xticks(x)
            ax1.set_xticklabels(tri_counts)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            speedups = [cpu_times[t] / gpu_times[t] if gpu_times.get(t, 0) > 0 else 0 for t in tri_counts]
            ax2.bar(x, speedups, color="#3498db")
            ax2.set_xlabel("Triangulos")
            ax2.set_ylabel("Speedup (CPU/GPU)")
            ax2.set_title(f"Speedup - {img_label}")
            ax2.set_xticks(x)
            ax2.set_xticklabels(tri_counts)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_dir = os.path.join(PROJECT_ROOT, "experiments", "plots", "gpu", img_label)
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"gpu_{img_label}_benchmark.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            print(f"Plot guardado: {plot_path}")

    except Exception as e:
        print(f"Error generando plot: {e}")

    print(f"\nResultados guardados en: {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
