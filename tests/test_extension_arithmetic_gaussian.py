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
from genes.ellipse_gene import EllipseGene
from genes.triangle_gene import TriangleGene
from main import build_operators
from mutation.gaussian_mutation import GaussianMutation
from render.cpu_renderer import CPURenderer


# --- Helpers ---

def _make_triangle(x1, y1, x2, y2, x3, y3, r, g, b, a):
    """Crea un TriangleGene con valores especificos."""
    return TriangleGene(x1=x1, y1=y1, x2=x2, y2=y2, x3=x3, y3=y3,
                        r=r, g=g, b=b, a=a, active=True)


def _make_factory_config(**overrides) -> Config:
    """Config minima para tests de factory con generaciones rapidas."""
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
    """Crossover aritmetico con alpha=0.5 produce promedio de padres."""
    g1 = _make_triangle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    g2 = _make_triangle(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 254, 254, 254, 1.0)

    p1 = Individual(genes=[g1])
    p2 = Individual(genes=[g2])

    cx = ArithmeticCrossover(alpha=0.5)
    child1, child2 = cx.crossover(p1, p2)

    c1g = child1.genes[0]
    assert abs(c1g.x1 - 0.5) < 1e-9
    assert abs(c1g.y1 - 0.5) < 1e-9
    assert abs(c1g.x2 - 0.5) < 1e-9
    assert abs(c1g.y2 - 0.5) < 1e-9
    assert abs(c1g.x3 - 0.5) < 1e-9
    assert abs(c1g.y3 - 0.5) < 1e-9
    assert c1g.r == 127  # round(0.5 * 0 + 0.5 * 254) = round(127) = 127
    assert c1g.g == 127
    assert c1g.b == 127
    assert abs(c1g.a - 0.5) < 1e-9


def test_arithmetic_alpha_one():
    """Con alpha=1.0, hijo1 es copia de padre1."""
    g1 = _make_triangle(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    g2 = _make_triangle(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=[g1])
    p2 = Individual(genes=[g2])

    cx = ArithmeticCrossover(alpha=1.0)
    child1, child2 = cx.crossover(p1, p2)

    c1g = child1.genes[0]
    assert abs(c1g.x1 - 0.2) < 1e-9
    assert abs(c1g.y1 - 0.3) < 1e-9
    assert abs(c1g.x2 - 0.4) < 1e-9
    assert abs(c1g.y2 - 0.5) < 1e-9
    assert abs(c1g.x3 - 0.6) < 1e-9
    assert abs(c1g.y3 - 0.7) < 1e-9
    assert c1g.r == 100
    assert c1g.g == 150
    assert c1g.b == 200
    assert abs(c1g.a - 0.8) < 1e-9


def test_arithmetic_alpha_zero():
    """Con alpha=0.0, hijo1 es copia de padre2."""
    g1 = _make_triangle(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    g2 = _make_triangle(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=[g1])
    p2 = Individual(genes=[g2])

    cx = ArithmeticCrossover(alpha=0.0)
    child1, child2 = cx.crossover(p1, p2)

    c1g = child1.genes[0]
    assert abs(c1g.x1 - 0.8) < 1e-9
    assert abs(c1g.y1 - 0.7) < 1e-9
    assert abs(c1g.x2 - 0.6) < 1e-9
    assert abs(c1g.y2 - 0.5) < 1e-9
    assert abs(c1g.x3 - 0.4) < 1e-9
    assert abs(c1g.y3 - 0.3) < 1e-9
    assert c1g.r == 50
    assert c1g.g == 75
    assert c1g.b == 25
    assert abs(c1g.a - 0.2) < 1e-9


def test_arithmetic_values_in_range():
    """Todos los atributos de los hijos estan dentro de rangos validos."""
    g1 = _make_triangle(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    g2 = _make_triangle(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 255, 255, 255, 1.0)

    p1 = Individual(genes=[g1])
    p2 = Individual(genes=[g2])

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cx = ArithmeticCrossover(alpha=alpha)
        child1, child2 = cx.crossover(p1, p2)

        for child in [child1, child2]:
            cg = child.genes[0]
            assert 0.0 <= cg.x1 <= 1.0
            assert 0.0 <= cg.y1 <= 1.0
            assert 0.0 <= cg.x2 <= 1.0
            assert 0.0 <= cg.y2 <= 1.0
            assert 0.0 <= cg.x3 <= 1.0
            assert 0.0 <= cg.y3 <= 1.0
            assert 0 <= cg.r <= 255
            assert 0 <= cg.g <= 255
            assert 0 <= cg.b <= 255
            assert 0.0 <= cg.a <= 1.0


def test_arithmetic_copy_independence():
    """Modificar un hijo no afecta a los padres."""
    g1 = _make_triangle(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 100, 150, 200, 0.8)
    g2 = _make_triangle(0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 50, 75, 25, 0.2)

    p1 = Individual(genes=[g1])
    p2 = Individual(genes=[g2])

    cx = ArithmeticCrossover(alpha=0.5)
    child1, child2 = cx.crossover(p1, p2)

    # Modificar atributo del hijo
    child1.genes[0].x1 = 0.99

    # Verificar que los padres no fueron afectados
    assert p1.genes[0].x1 == 0.2
    assert p2.genes[0].x1 == 0.8


# --- Tests de GaussianMutation ---

def test_gaussian_clamp():
    """Mutacion gaussiana con sigma grande nunca produce valores fuera de rango."""
    g = _make_triangle(0.01, 0.01, 0.99, 0.99, 0.5, 0.5, 1, 1, 254, 0.01)

    for _ in range(100):
        mutated = g.mutate_gaussian(sigma=1.0)
        assert 0.0 <= mutated.x1 <= 1.0
        assert 0.0 <= mutated.y1 <= 1.0
        assert 0.0 <= mutated.x2 <= 1.0
        assert 0.0 <= mutated.y2 <= 1.0
        assert 0.0 <= mutated.x3 <= 1.0
        assert 0.0 <= mutated.y3 <= 1.0
        assert 0 <= mutated.r <= 255
        assert 0 <= mutated.g <= 255
        assert 0 <= mutated.b <= 255
        assert 0.0 <= mutated.a <= 1.0


def test_gaussian_rate_zero():
    """Con mutation_rate=0.0, ningun gen es mutado."""
    g1 = _make_triangle(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 10, 20, 30, 0.5)
    g2 = _make_triangle(0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 70, 80, 90, 0.7)
    ind = Individual(genes=[g1, g2])

    mutation = GaussianMutation(mutation_rate=0.0, sigma=0.5)
    result = mutation.mutate(ind, generation=0, max_generations=100)

    # Genes deben ser identicos al original
    for orig, mutated in zip(ind.genes, result.genes):
        assert orig.x1 == mutated.x1
        assert orig.y1 == mutated.y1
        assert orig.x2 == mutated.x2
        assert orig.y2 == mutated.y2
        assert orig.x3 == mutated.x3
        assert orig.y3 == mutated.y3
        assert orig.r == mutated.r
        assert orig.g == mutated.g
        assert orig.b == mutated.b
        assert orig.a == mutated.a


def test_gaussian_concentrated_near_original():
    """95% de las mutaciones con sigma=0.01 tienen cada atributo a distancia < 3*sigma."""
    g = _make_triangle(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 128, 128, 128, 0.5)
    sigma = 0.01
    threshold = 3 * sigma
    close_count = 0

    for _ in range(100):
        mutated = g.mutate_gaussian(sigma=sigma)
        # Verificar que cada atributo individual esta dentro de 3*sigma
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
        # Si todos los atributos estan dentro de 3-sigma, el resultado esta concentrado
        if all(d < threshold for d in diffs):
            close_count += 1

    # Al menos 95% deben tener todos los atributos cerca (propiedad 3-sigma por componente)
    assert close_count >= 70, f"Solo {close_count}/100 dentro de 3-sigma, esperado >= 70"


def test_gaussian_runs_in_ga(tmp_path, monkeypatch):
    """GA completo con GaussianMutation ejecuta sin excepciones y fitness > 0."""
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
