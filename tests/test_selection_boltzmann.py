import warnings

import numpy as np

from core.individual import Individual
from core.population import Population
from selection.boltzmann import BoltzmannSelection


def _make_population(fitnesses: list[float]) -> Population:
    return Population(individuals=[Individual(genes=[], fitness=f) for f in fitnesses])


def test_boltzmann_high_temp_uniform():
    """Con temperatura muy alta, la seleccion es casi uniforme."""
    pop = _make_population([0.1, 0.3, 0.6, 0.9])
    selector = BoltzmannSelection(t0=1000.0, tc=1000.0, k=0.0)

    n_trials = 10000
    counts = [0] * 4
    fitnesses = [0.1, 0.3, 0.6, 0.9]

    np.random.seed(42)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        idx = fitnesses.index(selected[0].fitness)
        counts[idx] += 1

    for i in range(4):
        observed = counts[i] / n_trials
        assert abs(observed - 0.25) < 0.08, (
            f"Individual {i}: expected ~0.25, got {observed:.2f}"
        )


def test_boltzmann_low_temp_exploits():
    """Con temperatura muy baja, el mejor individuo domina la seleccion."""
    pop = _make_population([0.1, 0.3, 0.6, 0.9])
    selector = BoltzmannSelection(t0=0.01, tc=0.01, k=0.0)

    n_trials = 10000
    best_count = 0

    np.random.seed(42)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        if selected[0].fitness == 0.9:
            best_count += 1

    freq = best_count / n_trials
    assert freq > 0.90, f"Expected >90% selection of best but got {freq:.2%}"


def test_boltzmann_no_overflow():
    """Fitness cercanos a 1 con T baja no producen overflow."""
    pop = _make_population([0.99, 0.999, 0.9999, 0.99999])
    selector = BoltzmannSelection(t0=0.001, tc=0.001, k=0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = selector.select(pop, 4)

    assert len(result) == 4
    assert all(isinstance(ind.fitness, float) for ind in result)


def test_boltzmann_returns_k():
    """select() devuelve exactamente k individuos."""
    pop = _make_population([float(i + 1) / 10 for i in range(10)])
    selector = BoltzmannSelection(t0=100.0, tc=1.0, k=0.01)

    for k in [1, 5, 20]:
        result = selector.select(pop, k)
        assert len(result) == k
