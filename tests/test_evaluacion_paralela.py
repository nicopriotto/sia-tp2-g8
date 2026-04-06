import numpy as np
import pytest

from config.config_loader import Config
from core.genetic_algorithm import GeneticAlgorithm
from render.cpu_renderer import CPURenderer
from render.renderer import Renderer
from fitness.mse import MSEFitness
from selection.elite import EliteSelection
from crossover.one_point import OnePointCrossover
from mutation.gen_mutation import GenMutation
from survival.additive import AdditiveSurvival


def _make_config(**overrides):
    defaults = dict(
        triangle_count=3,
        population_size=10,
        max_generations=2,
        fitness_threshold=2.0,
        selection_method="Elite",
        crossover_methods=["OnePoint"],
        crossover_probability=0.7,
        mutation_methods=["Gen"],
        mutation_rate=0.1,
        survival_strategy="Aditiva",
        fitness_function="MSE",
        k_offspring=8,
        save_every=0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_ga(renderer=None, config=None):
    config = config or _make_config()
    target = np.random.rand(32, 32, 4).astype(np.float32)
    renderer = renderer or CPURenderer()
    return GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=MSEFitness(),
        selection=EliteSelection(),
        crossover_ops=[OnePointCrossover()],
        mutation_ops=[GenMutation(config.mutation_rate)],
        survival=AdditiveSurvival(),
    )


def test_evaluacion_paralela_produce_resultado():
    ga = _make_ga()
    result = ga.run()
    assert result.best_individual.fitness > 0


def test_gpu_renderer_no_usa_executor():
    class MockRenderer(Renderer):
        def render(self, genes, width, height):
            canvas = np.ones((height, width, 4), dtype=np.float32)
            return canvas

    ga = _make_ga(renderer=MockRenderer())
    assert ga._parallel is False


def test_cpu_renderer_usa_executor():
    ga = _make_ga(renderer=CPURenderer())
    assert ga._parallel is True
