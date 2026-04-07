import time

import numpy as np

from config.config_loader import Config
from core.ga_context import GAContext
from core.genetic_algorithm import EpochState, GeneticAlgorithm
from core.individual import Individual
from core.island_ga import IslandGeneticAlgorithm
from core.metrics_collector import MetricsCollector
from core.population import Population
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from mutation.gen_mutation import GenMutation
from render.cpu_renderer import CPURenderer
from selection.elite import EliteSelection
from survival.additive import AdditiveSurvival


def _make_config(**overrides):
    defaults = dict(
        triangle_count=3,
        population_size=6,
        max_generations=20,
        fitness_threshold=2.0,
        selection_method="Elite",
        crossover_methods=["OnePoint"],
        crossover_probability=0.0,
        mutation_methods=["Gen"],
        mutation_rate=0.1,
        survival_strategy="Aditiva",
        fitness_function="MSE",
        k_offspring=4,
        save_every=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_ga(config=None, output_dir="output"):
    config = config or _make_config()
    target = np.ones((20, 20, 4), dtype=np.float32)
    context = GAContext(generation=0, max_generations=config.max_generations)
    return GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=CPURenderer(),
        fitness_fn=MSEFitness(),
        selection=EliteSelection(),
        crossover_ops=[OnePointCrossover()],
        mutation_ops=[GenMutation(config.mutation_rate)],
        survival=AdditiveSurvival(),
        context=context,
        output_dir=output_dir,
    ), target


def test_run_epoch_executes_generations(tmp_path):
    """run_epoch ejecuta el rango de generaciones y actualiza el estado."""
    import os; os.chdir(tmp_path)
    ga, target = _make_ga()
    population = Population.random(6, 3)
    population.evaluate_all(target, ga.renderer, ga.fitness_fn)

    collector = MetricsCollector(
        output_dir=str(tmp_path), save_every=0,
        renderer=ga.renderer, width=20, height=20,
    )
    collector.init_csv()

    state = EpochState(
        population=population,
        best_fitness_history=[population.best.fitness],
    )
    start_time = time.time()
    state = ga.run_epoch(state, 1, 11, start_time, collector)

    assert state.last_generation == 10
    assert len(state.best_fitness_history) > 1
    assert state.population.best.fitness > 0


def test_run_still_works(tmp_path):
    """run() sigue retornando GAResult valido despues del refactor."""
    import os; os.chdir(tmp_path)
    ga, _ = _make_ga(config=_make_config(max_generations=10))
    result = ga.run()
    assert result.stop_reason == "generaciones_maximas"
    assert result.final_generation == 10
    assert result.best_individual.fitness > 0
    assert len(result.best_fitness_history) >= 10


def test_migrate_ring():
    """Migracion ring: mejor de isla 0 aparece en isla 1."""
    config = _make_config(
        island_enabled=True, island_count=3,
        island_migration_interval=10, island_migration_count=1,
    )

    # Crear 3 poblaciones con fitness controlados
    pops = []
    for i in range(3):
        inds = []
        for j in range(5):
            ind = Individual(genes=np.random.rand(3, 11), gene_type="triangle")
            ind.fitness = 0.9 if (i == 0 and j == 0) else 0.1
            inds.append(ind)
        pops.append(Population(individuals=inds))

    ga, target = _make_ga(config=config)
    island_ga = IslandGeneticAlgorithm(islands=[ga, ga, ga], config=config, target_image=target)
    island_ga._migrate_ring(pops, count=1)

    # El mejor de isla 0 (fitness=0.9) debe estar ahora en isla 1
    best_in_island_1 = max(pops[1].individuals, key=lambda x: x.fitness)
    assert best_in_island_1.fitness == 0.9


def test_migrate_fully_connected():
    """Migracion fully_connected: mejor de isla 0 aparece en islas 1 y 2."""
    config = _make_config(
        island_enabled=True, island_count=3,
        island_migration_interval=10, island_migration_count=1,
    )

    pops = []
    for i in range(3):
        inds = []
        for j in range(5):
            ind = Individual(genes=np.random.rand(3, 11), gene_type="triangle")
            ind.fitness = 0.9 if (i == 0 and j == 0) else 0.1
            inds.append(ind)
        pops.append(Population(individuals=inds))

    ga, target = _make_ga(config=config)
    island_ga = IslandGeneticAlgorithm(islands=[ga, ga, ga], config=config, target_image=target)
    island_ga._migrate_fully_connected(pops, count=1)

    best_in_1 = max(pops[1].individuals, key=lambda x: x.fitness)
    best_in_2 = max(pops[2].individuals, key=lambda x: x.fitness)
    assert best_in_1.fitness == 0.9
    assert best_in_2.fitness == 0.9


def test_island_ga_run_completes(tmp_path):
    """IslandGeneticAlgorithm.run() completa y retorna GAResult valido."""
    import os; os.chdir(tmp_path)
    config = _make_config(
        population_size=6, max_generations=30,
        island_enabled=True, island_count=3,
        island_migration_interval=10, island_migration_count=1,
        island_topology="ring",
    )

    islands = []
    target = np.ones((20, 20, 4), dtype=np.float32)
    for i in range(3):
        context = GAContext(generation=0, max_generations=config.max_generations)
        ga = GeneticAlgorithm(
            config=config,
            target_image=target,
            renderer=CPURenderer(),
            fitness_fn=MSEFitness(),
            selection=EliteSelection(),
            crossover_ops=[OnePointCrossover()],
            mutation_ops=[GenMutation(config.mutation_rate)],
            survival=AdditiveSurvival(),
            context=context,
            output_dir=str(tmp_path / f"island_{i}"),
        )
        islands.append(ga)

    island_ga = IslandGeneticAlgorithm(islands=islands, config=config, target_image=target)
    result = island_ga.run()

    assert result.best_individual.fitness > 0
    assert result.stop_reason != ""
    assert len(result.best_fitness_history) >= 1
    assert result.elapsed_seconds > 0
