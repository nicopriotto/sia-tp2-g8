import math
import random

import numpy as np
import pytest

from config.config_loader import Config
from core.ga_context import GAContext
from core.genetic_algorithm import GeneticAlgorithm
from core.individual import Individual
from crossover.arithmetic import ArithmeticCrossover
from fitness.mse import MSEFitness
from genes.triangle_gene import TriangleGene
from genes import gene_layout
from main import build_operators
from mutation.gaussian_mutation import GaussianMutation
from render.cpu_renderer import CPURenderer


def _make_triangle_row(x1, y1, x2, y2, x3, y3, r, g, b, a):
    return np.array([x1, y1, x2, y2, x3, y3, float(r), float(g), float(b), a, 1.0], dtype=np.float64)


def _make_factory_config(**overrides) -> Config:
    base = {
        "triangle_count": 10,
        "population_size": 10,
        "max_generations": 20,
        "fitness_threshold": 2.0,
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.7,
        "mutation_methods": ["Gaussiana"],
        "mutation_rate": 0.3,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
        "k_offspring": 6,
        "save_every": 0,
        "gaussian_sigma": 0.1,
    }
    base.update(overrides)
    return Config(**base)


# --- Tests de ArithmeticCrossover ---

def test_arithmetic_interpolation():
    r1 = _make_triangle_row(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    r2 = _make_triangle_row(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 254, 254, 254, 1.0)

    p1 = Individual(genes=r1.reshape(1, -1), gene_type="triangle")
    p2 = Individual(genes=r2.reshape(1, -1), gene_type="triangle")

    cx = ArithmeticCrossover(alpha=0.5)
    child1, child2 = cx.crossover(p1, p2)

    c1 = child1.genes[0]
    assert abs(c1[0] - 0.5) < 1e-9  # x1
    assert abs(c1[1] - 0.5) < 1e-9  # y1
    assert abs(c1[2] - 0.5) < 1e-9  # x2
    assert abs(c1[3] - 0.5) < 1e-9  # y2
    assert abs(c1[4] - 0.5) < 1e-9  # x3
    assert abs(c1[5] - 0.5) < 1e-9  # y3
    assert int(c1[6]) == 127  # r = round(127)
    assert int(c1[7]) == 127  # g
    assert int(c1[8]) == 127  # b
    assert abs(c1[9] - 0.5) < 1e-9  # a


def test_arithmetic_alpha_one():
    r1 = _make_triangle_row(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    r2 = _make_triangle_row(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=r1.reshape(1, -1), gene_type="triangle")
    p2 = Individual(genes=r2.reshape(1, -1), gene_type="triangle")

    cx = ArithmeticCrossover(alpha=1.0)
    child1, child2 = cx.crossover(p1, p2)

    c1 = child1.genes[0]
    assert abs(c1[0] - 0.2) < 1e-9
    assert abs(c1[1] - 0.3) < 1e-9
    assert int(c1[6]) == 100
    assert int(c1[7]) == 150
    assert int(c1[8]) == 200
    assert abs(c1[9] - 0.8) < 1e-9


def test_arithmetic_alpha_zero():
    r1 = _make_triangle_row(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    r2 = _make_triangle_row(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=r1.reshape(1, -1), gene_type="triangle")
    p2 = Individual(genes=r2.reshape(1, -1), gene_type="triangle")

    cx = ArithmeticCrossover(alpha=0.0)
    child1, child2 = cx.crossover(p1, p2)

    c1 = child1.genes[0]
    assert abs(c1[0] - 0.8) < 1e-9
    assert int(c1[6]) == 50
    assert int(c1[7]) == 75
    assert abs(c1[9] - 0.2) < 1e-9


def test_arithmetic_values_in_range():
    r1 = _make_triangle_row(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    r2 = _make_triangle_row(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 255, 255, 255, 1.0)

    p1 = Individual(genes=r1.reshape(1, -1), gene_type="triangle")
    p2 = Individual(genes=r2.reshape(1, -1), gene_type="triangle")

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cx = ArithmeticCrossover(alpha=alpha)
        child1, child2 = cx.crossover(p1, p2)

        for child in [child1, child2]:
            row = child.genes[0]
            for i in range(6):
                assert 0.0 <= row[i] <= 1.0
            assert 0 <= row[6] <= 255
            assert 0 <= row[7] <= 255
            assert 0 <= row[8] <= 255
            assert 0.0 <= row[9] <= 1.0


def test_arithmetic_copy_independence():
    r1 = _make_triangle_row(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    r2 = _make_triangle_row(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=r1.reshape(1, -1), gene_type="triangle")
    p2 = Individual(genes=r2.reshape(1, -1), gene_type="triangle")

    cx = ArithmeticCrossover(alpha=0.5)
    child1, child2 = cx.crossover(p1, p2)

    child1.genes[0, 0] = 0.99

    assert p1.genes[0, 0] == 0.2
    assert p2.genes[0, 0] == 0.8


# --- Tests de GaussianMutation ---

def test_gaussian_clamp():
    g = TriangleGene(x1=0.01, y1=0.01, x2=0.99, y2=0.99, x3=0.5, y3=0.5, r=1, g=1, b=254, a=0.01)
    for _ in range(100):
        mutated = g.mutate_gaussian(sigma=1.0)
        assert 0.0 <= mutated.x1 <= 1.0
        assert 0 <= mutated.r <= 255
        assert 0.0 <= mutated.a <= 1.0


def test_gaussian_rate_zero():
    np.random.seed(42)
    r1 = _make_triangle_row(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10, 20, 30, 0.5)
    r2 = _make_triangle_row(0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 70, 80, 90, 0.7)
    genes = np.vstack([r1, r2])
    ind = Individual(genes=genes, gene_type="triangle")

    mutation = GaussianMutation(mutation_rate=0.0, sigma=0.5)
    result = mutation.mutate(ind, generation=0, max_generations=100)

    np.testing.assert_array_equal(ind.genes, result.genes)


def test_gaussian_concentrated_near_original():
    g = TriangleGene(x1=0.5, y1=0.5, x2=0.5, y2=0.5, x3=0.5, y3=0.5, r=128, g=128, b=128, a=0.5)
    sigma = 0.01
    threshold = 3 * sigma
    close_count = 0

    for _ in range(100):
        mutated = g.mutate_gaussian(sigma=sigma)
        diffs = [
            abs(mutated.x1 - g.x1),
            abs(mutated.y1 - g.y1),
            abs(mutated.x2 - g.x2),
            abs(mutated.y2 - g.y2),
            abs(mutated.x3 - g.x3),
            abs(mutated.y3 - g.y3),
            abs(mutated.r - g.r) / 255.0,
            abs(mutated.g - g.g) / 255.0,
            abs(mutated.b - g.b) / 255.0,
            abs(mutated.a - g.a),
        ]
        if all(d < threshold for d in diffs):
            close_count += 1

    assert close_count >= 70, f"Solo {close_count}/100 dentro de 3-sigma, esperado >= 70"


def test_gaussian_runs_in_ga(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = np.random.rand(10, 10, 4).astype(np.float32)
    renderer = CPURenderer()

    config = _make_factory_config()

    selection_ops, crossover_ops, mutation_ops, survival, fitness = build_operators(config)
    context = GAContext(generation=0, max_generations=config.max_generations)

    ga = GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=fitness,
        selection_ops=selection_ops,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
        context=context,
    )

    result = ga.run()
    assert result.best_individual is not None
    assert result.best_individual.fitness > 0
