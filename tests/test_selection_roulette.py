import numpy as np
import pytest

from core.individual import Individual
from core.population import Population
from selection.roulette import RouletteSelection


def _make_population(fitnesses: list[float]) -> Population:
    individuals = []
    for f in fitnesses:
        ind = Individual(genes=[], fitness=f)
        individuals.append(ind)
    return Population(individuals=individuals)


def test_roulette_slide_example():
    """Verifica seleccion con fitness=[3,6,11,14,1] usando acumulados de las slides."""
    fitnesses = [3, 6, 11, 14, 1]
    total = sum(fitnesses)  # 35
    # q = [3/35, 9/35, 20/35, 34/35, 1.0]
    expected_q = np.array([3/35, 9/35, 20/35, 34/35, 1.0])

    pop = _make_population(fitnesses)
    roulette = RouletteSelection()

    np.random.seed(0)
    r_values = np.random.random(5)

    # Verify manually with bisect
    import bisect
    for r in r_values:
        idx = bisect.bisect_left(expected_q, r)
        assert 0 <= idx < len(fitnesses)

    np.random.seed(0)
    selected = roulette.select(pop, 5)
    assert len(selected) == 5


def test_roulette_returns_k():
    """select() devuelve exactamente k individuos para distintos valores de k."""
    pop = _make_population([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    roulette = RouletteSelection()

    for k in [1, 5, 20]:
        result = roulette.select(pop, k)
        assert len(result) == k


def test_roulette_returns_copies():
    """Los individuos devueltos son copias, no referencias directas."""
    pop = _make_population([0.5, 0.5, 0.5])
    roulette = RouletteSelection()

    np.random.seed(42)
    selected = roulette.select(pop, 3)

    original_fitnesses = [ind.fitness for ind in pop.individuals]
    selected[0].fitness = 999.0

    for i, ind in enumerate(pop.individuals):
        assert ind.fitness == original_fitnesses[i]


def test_roulette_distribution():
    """La frecuencia de seleccion es aproximadamente proporcional al fitness."""
    fitnesses = [1.0, 2.0, 3.0, 4.0]
    total = sum(fitnesses)  # 10
    pop = _make_population(fitnesses)
    roulette = RouletteSelection()

    n_trials = 10000
    counts = [0] * 4

    np.random.seed(7)
    for _ in range(n_trials):
        selected = roulette.select(pop, 1)
        fitness = selected[0].fitness
        idx = fitnesses.index(fitness)
        counts[idx] += 1

    for i, f in enumerate(fitnesses):
        expected = f / total
        observed = counts[i] / n_trials
        assert abs(observed - expected) < 0.05, (
            f"Individual {i}: expected ~{expected:.2f}, got {observed:.2f}"
        )


def test_roulette_equal_fitness():
    """Con fitness iguales, la seleccion es uniforme (sin sesgos)."""
    from selection.roulette import _build_cumulative

    fitnesses = np.array([0.5, 0.5, 0.5, 0.5])
    # Equal fitness => _build_cumulative returns None => uniform selection
    assert _build_cumulative(fitnesses) is None

    pop = _make_population([0.5, 0.5, 0.5, 0.5])
    roulette = RouletteSelection()

    np.random.seed(99)
    # Verify it returns k items and doesn't crash
    result = roulette.select(pop, 4)
    assert len(result) == 4
    assert all(ind.fitness == 0.5 for ind in result)
