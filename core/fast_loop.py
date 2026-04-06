"""Fast evolutionary loop that operates on raw numpy arrays instead of Python objects.

Population: np.ndarray shape (pop_size, n_genes, 11)
Fitness: np.ndarray shape (pop_size,)

Gene layout per row (11 floats):
  [0-5]: x1, y1, x2, y2, x3, y3  (coords in [0,1])
  [6-8]: r, g, b                   (color in [0,255])
  [9]:   alpha                     (in [0,1])
  [10]:  active                    (1.0 or 0.0)

All crossover/mutation/selection operations are vectorized numpy ops.
"""

import logging
import time

import numpy as np

from config.config_loader import Config
from core.ga_context import GAContext
from core.ga_result import GAResult
from core.individual import Individual
from core.metrics_collector import MetricsCollector
from genes.triangle_gene import TriangleGene

logger = logging.getLogger(__name__)


def _random_population(pop_size: int, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    """Create random population array (pop_size, n_genes, 11)."""
    pop = np.zeros((pop_size, n_genes, 11), dtype=np.float32)
    pop[:, :, 0:6] = rng.random((pop_size, n_genes, 6), dtype=np.float32)  # coords
    pop[:, :, 6:9] = rng.integers(0, 256, size=(pop_size, n_genes, 3)).astype(np.float32)  # rgb
    pop[:, :, 9] = rng.random((pop_size, n_genes), dtype=np.float32)  # alpha
    pop[:, :, 10] = 1.0  # active
    return pop


def _elite_select(pop: np.ndarray, fitness: np.ndarray, k: int) -> np.ndarray:
    """Select k individuals by elite selection. Returns indices."""
    sorted_idx = np.argsort(fitness)[::-1]  # descending
    n = len(fitness)
    if k <= n:
        return sorted_idx[:k]
    # Need repetitions
    reps = (k + n - 1) // n
    repeated = np.tile(sorted_idx, reps)
    return repeated[:k]


def _boltzmann_select(pop: np.ndarray, fitness: np.ndarray, k: int,
                       generation: int, t0: float, tc: float, bk: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Boltzmann selection returning k indices."""
    temp = tc + (t0 - tc) * np.exp(-bk * generation)
    temp = max(temp, 1e-6)
    shifted = (fitness - np.max(fitness)) / temp
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)
    return rng.choice(len(fitness), size=k, p=probs)


def _crossover_one_point(parents: np.ndarray, pc: float, rng: np.random.Generator) -> np.ndarray:
    """One-point crossover on array (k, n_genes, 11). Operates in pairs."""
    k, n_genes, attrs = parents.shape
    children = parents.copy()
    n_pairs = k // 2

    # Which pairs actually cross over
    do_cross = rng.random(n_pairs) < pc
    cut_points = rng.integers(0, n_genes, size=n_pairs)

    for i in range(n_pairs):
        if do_cross[i]:
            p = cut_points[i]
            idx1 = i * 2
            idx2 = i * 2 + 1
            children[idx1, p:] = parents[idx2, p:]
            children[idx2, p:] = parents[idx1, p:]

    return children


def _crossover_two_point(parents: np.ndarray, pc: float, rng: np.random.Generator) -> np.ndarray:
    """Two-point crossover."""
    k, n_genes, attrs = parents.shape
    children = parents.copy()
    n_pairs = k // 2

    do_cross = rng.random(n_pairs) < pc
    points = np.sort(rng.integers(0, n_genes, size=(n_pairs, 2)), axis=1)

    for i in range(n_pairs):
        if do_cross[i]:
            p1, p2 = points[i]
            idx1 = i * 2
            idx2 = i * 2 + 1
            children[idx1, p1:p2] = parents[idx2, p1:p2]
            children[idx2, p1:p2] = parents[idx1, p1:p2]

    return children


def _crossover_uniform(parents: np.ndarray, pc: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover."""
    k, n_genes, attrs = parents.shape
    children = parents.copy()
    n_pairs = k // 2

    do_cross = rng.random(n_pairs) < pc

    for i in range(n_pairs):
        if do_cross[i]:
            idx1 = i * 2
            idx2 = i * 2 + 1
            mask = rng.random(n_genes) < 0.5
            children[idx1, mask] = parents[idx2, mask]
            children[idx2, mask] = parents[idx1, mask]

    return children


def _mutate_gen(children: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Gen mutation: replace 1 random gene with probability mutation_rate."""
    k, n_genes, _ = children.shape
    do_mutate = rng.random(k) < mutation_rate
    indices = np.where(do_mutate)[0]

    if len(indices) > 0:
        gene_idx = rng.integers(0, n_genes, size=len(indices))
        for i, gi in zip(indices, gene_idx):
            children[i, gi, 0:6] = rng.random(6, dtype=np.float32)
            children[i, gi, 6:9] = rng.integers(0, 256, size=3).astype(np.float32)
            children[i, gi, 9] = rng.random(dtype=np.float32)
            children[i, gi, 10] = 1.0

    return children


def _mutate_multigen(children: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """MultiGen mutation: replace 1..M random genes with probability mutation_rate."""
    k, n_genes, _ = children.shape
    do_mutate = rng.random(k) < mutation_rate
    indices = np.where(do_mutate)[0]

    if len(indices) > 0:
        max_m = max(1, n_genes // 4)
        for i in indices:
            m = rng.integers(1, max_m + 1)
            gene_indices = rng.choice(n_genes, size=m, replace=False)
            for gi in gene_indices:
                children[i, gi, 0:6] = rng.random(6, dtype=np.float32)
                children[i, gi, 6:9] = rng.integers(0, 256, size=3).astype(np.float32)
                children[i, gi, 9] = rng.random(dtype=np.float32)
                children[i, gi, 10] = 1.0

    return children


def _mutate_non_uniform(children: np.ndarray, mutation_rate: float,
                         generation: int, max_generations: int,
                         b: float, rng: np.random.Generator) -> np.ndarray:
    """Non-uniform mutation: delta decreases with generation."""
    k, n_genes, _ = children.shape
    do_mutate = rng.random(k) < mutation_rate
    indices = np.where(do_mutate)[0]

    if len(indices) > 0:
        tau = generation / max(max_generations, 1)
        strength = (1.0 - tau ** b)

        gene_idx = rng.integers(0, n_genes, size=len(indices))
        for i, gi in zip(indices, gene_idx):
            # Delta on coords and alpha
            delta_coords = rng.uniform(-strength, strength, size=6).astype(np.float32)
            children[i, gi, 0:6] = np.clip(children[i, gi, 0:6] + delta_coords, 0.0, 1.0)

            delta_rgb = rng.uniform(-strength * 255, strength * 255, size=3).astype(np.float32)
            children[i, gi, 6:9] = np.clip(children[i, gi, 6:9] + delta_rgb, 0.0, 255.0)

            delta_a = rng.uniform(-strength, strength)
            children[i, gi, 9] = np.clip(children[i, gi, 9] + delta_a, 0.0, 1.0)

    return children


def _array_to_individual(genes_array: np.ndarray) -> Individual:
    """Convert a single individual's gene array (n_genes, 11) to Individual object."""
    genes = []
    for row in genes_array:
        genes.append(TriangleGene(
            x1=float(row[0]), y1=float(row[1]),
            x2=float(row[2]), y2=float(row[3]),
            x3=float(row[4]), y3=float(row[5]),
            r=int(row[6]), g=int(row[7]), b=int(row[8]),
            a=float(row[9]),
            active=bool(row[10] > 0.5),
        ))
    return Individual(genes=genes, fitness=0.0)


def _choose_crossover(methods: list[str], weights: list[float], rng: np.random.Generator):
    """Choose crossover function based on config."""
    name_map = {
        "OnePoint": _crossover_one_point,
        "TwoPoint": _crossover_two_point,
        "Uniform": _crossover_uniform,
    }
    # Weighted choice
    idx = rng.choice(len(methods), p=weights)
    method = methods[idx]
    return name_map.get(method, _crossover_one_point)


def _choose_mutation(methods: list[str], weights: list[float], rng: np.random.Generator):
    """Choose mutation function based on config."""
    idx = rng.choice(len(methods), p=weights)
    return methods[idx]


def run_fast(
    config: Config,
    target_image: np.ndarray,
    renderer,
    fitness_fn,
    output_dir: str = "output",
    context: GAContext | None = None,
) -> GAResult:
    """Fast evolutionary loop using numpy arrays instead of Python objects."""
    height, width = target_image.shape[0], target_image.shape[1]
    rng = np.random.default_rng()

    collector = MetricsCollector(
        output_dir=output_dir,
        save_every=config.save_every,
        renderer=renderer,
        width=width,
        height=height,
    )
    collector.init_csv()

    start_time = time.time()

    n_genes = config.triangle_count
    pop_size = config.population_size

    # Initialize population as numpy array
    pop = _random_population(pop_size, n_genes, rng)
    fitness = np.array(renderer.evaluate_batch_raw(pop), dtype=np.float32)

    best_idx = np.argmax(fitness)
    best_fitness = fitness[best_idx]

    # For snapshots, convert best to Individual
    best_individual = _array_to_individual(pop[best_idx])
    best_individual.fitness = float(best_fitness)

    avg_fitness = float(np.mean(fitness))
    std_fitness = float(np.std(fitness))

    # Log initial population via a lightweight Population-like interface
    _log_and_snapshot(collector, 0, best_individual, best_fitness, avg_fitness, std_fitness,
                      time.time() - start_time)

    logger.info(
        "Poblacion inicial | Best: %.4f | Avg: %.4f | Std: %.4f",
        best_fitness, avg_fitness, std_fitness,
    )

    k_offspring = max(2, int(config.generational_gap * pop_size))
    if k_offspring % 2 != 0:
        k_offspring += 1

    # Normalize weights
    cx_methods = config.crossover_methods
    cx_weights = np.array(config.crossover_weights if config.crossover_weights else [1.0] * len(cx_methods))
    cx_weights = cx_weights / cx_weights.sum()

    mut_methods = config.mutation_methods
    mut_weights = np.array(config.mutation_weights if config.mutation_weights else [1.0] * len(mut_methods))
    mut_weights = mut_weights / mut_weights.sum()

    sel_method = config.selection_method
    mutation_rate = config.mutation_rate

    stop_reason = None
    last_generation = 0
    best_fitness_history = [float(best_fitness)]
    low_diversity_counter = 0

    for generation in range(1, config.max_generations + 1):
        generation_start = time.time()

        if context is not None:
            context.generation = generation

        # Time cutoff
        if config.max_seconds > 0:
            if time.time() - start_time >= config.max_seconds:
                stop_reason = "tiempo_maximo"
                last_generation = generation - 1
                break

        # Selection (returns indices into pop)
        if sel_method == "Boltzmann":
            parent_idx = _boltzmann_select(
                pop, fitness, k_offspring, generation,
                config.boltzmann_t0, config.boltzmann_tc, config.boltzmann_k, rng,
            )
        else:
            parent_idx = _elite_select(pop, fitness, k_offspring)

        parents = pop[parent_idx].copy()

        # Crossover
        cx_fn = _choose_crossover(cx_methods, cx_weights, rng)
        children = cx_fn(parents, config.crossover_probability, rng)

        # Mutation
        mut_name = _choose_mutation(mut_methods, mut_weights, rng)
        if mut_name == "NoUniforme":
            children = _mutate_non_uniform(children, mutation_rate, generation,
                                            config.max_generations, config.non_uniform_b, rng)
        elif mut_name == "MultiGen":
            children = _mutate_multigen(children, mutation_rate, rng)
        else:  # Gen, default
            children = _mutate_gen(children, mutation_rate, rng)

        # Evaluate children fitness on GPU
        children_fitness = np.array(renderer.evaluate_batch_raw(children), dtype=np.float32)

        # Survival: combine and select best N
        combined_pop = np.concatenate([pop, children], axis=0)
        combined_fitness = np.concatenate([fitness, children_fitness])

        if sel_method == "Boltzmann":
            surv_idx = _boltzmann_select(
                combined_pop, combined_fitness, pop_size, generation,
                config.boltzmann_t0, config.boltzmann_tc, config.boltzmann_k, rng,
            )
        else:
            surv_idx = _elite_select(combined_pop, combined_fitness, pop_size)

        pop = combined_pop[surv_idx].copy()
        fitness = combined_fitness[surv_idx].copy()

        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        avg_fitness = float(np.mean(fitness))
        std_fitness = float(np.std(fitness))

        best_fitness_history.append(float(best_fitness))

        elapsed = time.time() - start_time

        # Snapshot only when needed (expensive: creates objects)
        need_snapshot = (config.save_every > 0 and generation % config.save_every == 0)
        if need_snapshot:
            best_individual = _array_to_individual(pop[best_idx])
            best_individual.fitness = float(best_fitness)
            _log_and_snapshot(collector, generation, best_individual, best_fitness,
                              avg_fitness, std_fitness, elapsed)
        else:
            _log_metrics_only(collector, generation, best_fitness, avg_fitness, std_fitness, elapsed)

        generation_time = time.time() - generation_start
        logger.info(
            "Gen %d | Best: %.4f | Avg: %.4f | Std: %.4f | Time: %.3fs",
            generation, best_fitness, avg_fitness, std_fitness, generation_time,
        )

        last_generation = generation

        # Stop criteria
        if best_fitness >= config.fitness_threshold:
            stop_reason = "fitness_alcanzado"
            break

        if config.min_error > 0:
            current_error = (1.0 / best_fitness) - 1.0 if best_fitness > 0 else float('inf')
            if current_error <= config.min_error:
                stop_reason = "error_minimo"
                break

        if config.content_generations > 0 and len(best_fitness_history) >= config.content_generations:
            recent = best_fitness_history[-config.content_generations:]
            if max(recent) - min(recent) < config.content_threshold:
                stop_reason = "contenido"
                break

        if config.structure_generations > 0:
            if std_fitness < config.structure_threshold:
                low_diversity_counter += 1
            else:
                low_diversity_counter = 0
            if low_diversity_counter >= config.structure_generations:
                stop_reason = "estructura"
                break

        if generation >= config.max_generations:
            stop_reason = "generaciones_maximas"
            break

    if stop_reason is None:
        stop_reason = "generaciones_maximas"

    elapsed_seconds = time.time() - start_time

    # Final output
    best_idx = np.argmax(fitness)
    best_individual = _array_to_individual(pop[best_idx])
    best_individual.fitness = float(fitness[best_idx])

    collector.save_final_result(best_individual, config, last_generation)
    collector.save_final_image(best_individual)

    logger.info("=== FIN ===")
    logger.info("Motivo de corte: %s", stop_reason)
    logger.info("Mejor fitness final: %.4f", fitness[best_idx])
    logger.info("Fitness promedio final: %.4f", np.mean(fitness))

    return GAResult(
        best_individual=best_individual,
        final_generation=last_generation,
        stop_reason=stop_reason,
        elapsed_seconds=elapsed_seconds,
        best_fitness_history=best_fitness_history,
    )


def _log_and_snapshot(collector, generation, best_individual, best_fit, avg_fit, std_fit, elapsed):
    """Log metrics and save snapshot image."""
    # Write CSV row directly
    import csv
    with open(f"{collector.output_dir}/metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([generation, best_fit, avg_fit, std_fit, round(elapsed, 3)])

    # Save snapshot
    if collector.save_every > 0 and generation % collector.save_every == 0:
        from PIL import Image as PILImage
        image = collector.renderer.render(best_individual.genes, collector.width, collector.height)
        img_uint8 = (image * 255).astype(np.uint8)
        PILImage.fromarray(img_uint8).save(f"{collector.output_dir}/gen_{generation:04d}.png")


def _log_metrics_only(collector, generation, best_fit, avg_fit, std_fit, elapsed):
    """Log metrics CSV row without creating Individual objects."""
    import csv
    with open(f"{collector.output_dir}/metrics.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([generation, best_fit, avg_fit, std_fit, round(elapsed, 3)])
