import numpy as np

from core.individual import Individual
from core.population import Population
from selection.ranking import RankingSelection


def _make_population(fitnesses: list[float]) -> Population:
    return Population(individuals=[Individual(genes=[], fitness=f) for f in fitnesses])


def test_ranking_reduces_pressure():
    """El individuo dominante (fitness=1000) es seleccionado mucho menos que con Ruleta."""
    pop = _make_population([1000.0, 1.0, 1.0, 1.0])
    selector = RankingSelection()

    n_trials = 10000
    best_count = 0

    np.random.seed(11)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        if selected[0].fitness == 1000.0:
            best_count += 1

    freq = best_count / n_trials
    # With roulette it'd be ~99.7%, ranking gives ~50% — assert well below 80%
    assert freq < 0.80, f"Expected < 80% but got {freq:.2%}"


def test_ranking_returns_k():
    """select() devuelve exactamente k individuos."""
    pop = _make_population([float(i + 1) for i in range(10)])
    selector = RankingSelection()

    for k in [1, 5, 20]:
        result = selector.select(pop, k)
        assert len(result) == k


def test_ranking_pseudo_fitness():
    """El peor individuo (rank=N) nunca es seleccionado (pseudo-fitness=0)."""
    fitnesses = [10.0, 8.0, 6.0, 4.0, 2.0]
    pop = _make_population(fitnesses)
    selector = RankingSelection()

    n_trials = 5000
    worst_count = 0

    np.random.seed(7)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        if selected[0].fitness == 2.0:
            worst_count += 1

    assert worst_count == 0, f"Worst individual should never be selected but was {worst_count} times"
