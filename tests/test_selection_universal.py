from unittest.mock import patch

import numpy as np

from core.individual import Individual
from core.population import Population
from selection.roulette import RouletteSelection
from selection.universal import UniversalSelection


def _make_population(fitnesses: list[float]) -> Population:
    return Population(individuals=[Individual(genes=[], fitness=f) for f in fitnesses])


def test_universal_slide_example():
    """K=4, r=0.084 => pointers [0.021, 0.271, 0.521, 0.771]."""
    # fitness=[1,2,3,4] total=10 => q=[0.1, 0.3, 0.6, 1.0]
    # pointers: 0.021->idx0, 0.271->idx1, 0.521->idx2, 0.771->idx3
    pop = _make_population([1.0, 2.0, 3.0, 4.0])
    selector = UniversalSelection()

    with patch("selection.universal.np.random.uniform", return_value=0.084 / 4):
        selected = selector.select(pop, 4)

    assert len(selected) == 4
    fitnesses = [ind.fitness for ind in selected]
    assert fitnesses == [1.0, 2.0, 3.0, 4.0]


def test_universal_returns_k():
    """select() devuelve exactamente k individuos."""
    pop = _make_population([float(i + 1) for i in range(10)])
    selector = UniversalSelection()

    for k in [1, 5, 20]:
        result = selector.select(pop, k)
        assert len(result) == k


def test_universal_lower_variance():
    """Universal tiene menor varianza de muestreo que Ruleta."""
    fitnesses = [1.0, 2.0, 3.0, 4.0, 5.0]
    pop = _make_population(fitnesses)
    roulette = RouletteSelection()
    universal = UniversalSelection()

    n_runs = 1000
    k = 5

    roulette_counts = []
    universal_counts = []

    np.random.seed(42)
    for _ in range(n_runs):
        selected = roulette.select(pop, k)
        counts = [sum(1 for s in selected if s.fitness == f) for f in fitnesses]
        roulette_counts.append(counts)

    np.random.seed(42)
    for _ in range(n_runs):
        selected = universal.select(pop, k)
        counts = [sum(1 for s in selected if s.fitness == f) for f in fitnesses]
        universal_counts.append(counts)

    roulette_var = np.var(roulette_counts)
    universal_var = np.var(universal_counts)

    assert universal_var < roulette_var, (
        f"Universal variance {universal_var:.4f} should be less than Roulette variance {roulette_var:.4f}"
    )
