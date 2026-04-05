import numpy as np

from config.config_loader import Config
from core.genetic_algorithm import GeneticAlgorithm
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from mutation.gen_mutation import GenMutation
from render.cpu_renderer import CPURenderer
from selection.elite import EliteSelection
from survival.additive import AdditiveSurvival


def _make_ga(tmp_path, **overrides):
    defaults = dict(
        triangle_count=3,
        population_size=6,
        max_generations=10,
        fitness_threshold=2.0,
        selection_method="elite",
        crossover_methods=["one_point"],
        crossover_probability=0.0,
        mutation_methods=["gen"],
        mutation_rate=0.1,
        survival_strategy="generational",
        fitness_function="mse",
        k_offspring=4,
        save_every=0,
    )
    defaults.update(overrides)
    config = Config(**defaults)
    target = np.ones((20, 20, 4), dtype=np.float32)
    import os; os.chdir(tmp_path)
    return GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=CPURenderer(),
        fitness_fn=MSEFitness(),
        selection=EliteSelection(),
        crossover_ops=[OnePointCrossover()],
        mutation_ops=[GenMutation(config.mutation_rate)],
        survival=AdditiveSurvival(),
    )


def _calc_k_offspring(generational_gap: float, population_size: int) -> int:
    k = max(2, int(generational_gap * population_size))
    if k % 2 != 0:
        k += 1
    return k


def test_gap_g1_generates_n_children():
    k = _calc_k_offspring(1.0, 20)
    assert k == 20


def test_gap_g05_generates_half_children():
    k = _calc_k_offspring(0.5, 20)
    assert k == 10


def test_gap_minimum_two_children():
    k = _calc_k_offspring(0.01, 20)
    assert k >= 2


def test_stop_max_generations(tmp_path):
    ga = _make_ga(tmp_path, max_generations=10, fitness_threshold=2.0, max_seconds=0)
    result = ga.run()
    assert result.stop_reason == "generaciones_maximas"
    assert result.final_generation == 10


def test_stop_fitness_threshold(tmp_path):
    ga = _make_ga(tmp_path, max_generations=1000, fitness_threshold=0.0)
    result = ga.run()
    assert result.stop_reason == "fitness_alcanzado"
    assert result.final_generation < 1000


def test_stop_by_time(tmp_path):
    ga = _make_ga(tmp_path, max_generations=999999, fitness_threshold=2.0, max_seconds=1)
    result = ga.run()
    assert result.stop_reason == "tiempo_maximo"
    assert 0.8 <= result.elapsed_seconds <= 5.0


def test_stop_by_content(tmp_path):
    ga = _make_ga(
        tmp_path,
        max_generations=1000,
        fitness_threshold=2.0,
        content_generations=5,
        content_threshold=1.0,  # very high threshold — any stable fitness triggers it
    )
    result = ga.run()
    assert result.stop_reason == "contenido"


def test_stop_reason_logged(tmp_path):
    ga = _make_ga(tmp_path, max_generations=5)
    result = ga.run()
    assert result.stop_reason != ""
    assert len(result.best_fitness_history) >= 5
