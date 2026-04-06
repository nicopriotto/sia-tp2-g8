import numpy as np

from config.config_loader import Config
from core.genetic_algorithm import GeneticAlgorithm
from core.individual import Individual
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from genes.triangle_gene import TriangleGene
from genes import gene_layout
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


def test_inactive_gene_not_rendered():
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=1.0, active=False,
    )
    genes = gene.to_row().reshape(1, -1)
    renderer = CPURenderer()
    result = renderer.render(genes, 50, 50)

    expected = np.ones((50, 50, 4), dtype=np.float32)
    np.testing.assert_array_almost_equal(result, expected)


def test_active_gene_rendered():
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=1.0, active=True,
    )
    genes = gene.to_row().reshape(1, -1)
    renderer = CPURenderer()
    result = renderer.render(genes, 50, 50)

    red_pixels = (result[:, :, 0] > 0.9) & (result[:, :, 1] < 0.1) & (result[:, :, 2] < 0.1)
    assert np.any(red_pixels), "Deberia haber pixeles rojos renderizados"


def test_min_error_stop_fires(tmp_path):
    ga = _make_ga(tmp_path, max_generations=500, min_error=1.0, fitness_threshold=2.0)
    result = ga.run()
    assert result.stop_reason == "error_minimo"
    assert result.final_generation < 500


def test_min_error_zero_disabled(tmp_path):
    ga = _make_ga(tmp_path, max_generations=5, min_error=0.0, fitness_threshold=2.0)
    result = ga.run()
    assert result.stop_reason == "generaciones_maximas"
    assert result.final_generation == 5


def test_individual_with_inactive_genes_valid():
    genes = gene_layout.random_genes("triangle", 10)
    genes[:5, 10] = 1.0   # 5 activos
    genes[5:, 10] = 0.0   # 5 inactivos

    individual = Individual(genes=genes, gene_type="triangle")
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    target = np.ones((20, 20, 4), dtype=np.float32)

    individual.compute_fitness(target, renderer, fitness_fn)

    assert individual.fitness > 0.0
    assert individual.fitness <= 1.0
