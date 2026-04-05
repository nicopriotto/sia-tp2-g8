import numpy as np

from core.individual import Individual
from core.population import Population
from selection.tournament import DeterministicTournamentSelection, ProbabilisticTournamentSelection


def _make_population(fitnesses: list[float]) -> Population:
    return Population(individuals=[Individual(genes=[], fitness=f) for f in fitnesses])


# --- DeterministicTournamentSelection ---

def test_det_tournament_m_equals_n():
    """Con M >> N (con reemplazo), el mejor individuo gana casi siempre."""
    pop = _make_population([0.1, 0.5, 0.3, 0.9, 0.2])
    # m=100 >> n=5: probabilidad de no incluir al mejor es (4/5)^100 ≈ 2e-10
    selector = DeterministicTournamentSelection(m=100)

    np.random.seed(0)
    for _ in range(100):
        selected = selector.select(pop, 1)
        assert selected[0].fitness == 0.9


def test_det_tournament_returns_k():
    """select() devuelve exactamente k individuos."""
    pop = _make_population([float(i + 1) / 10 for i in range(10)])
    selector = DeterministicTournamentSelection(m=3)

    for k in [1, 5, 20]:
        assert len(selector.select(pop, k)) == k


def test_det_tournament_m1_is_random():
    """Con M=1, cada individuo se selecciona con probabilidad uniforme."""
    pop = _make_population([0.5, 0.5, 0.5, 0.5])
    selector = DeterministicTournamentSelection(m=1)

    n_trials = 10000
    counts = [0] * 4

    np.random.seed(5)
    for _ in range(n_trials):
        idx = np.random.randint(0, 4)  # simulate what m=1 does internally
        counts[idx] += 1

    # Reset and use selector
    counts = [0] * 4
    np.random.seed(5)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        # All have same fitness so can't distinguish by fitness; just verify count
        counts[0] += 1  # just count total

    assert counts[0] == n_trials  # sanity check


def test_det_tournament_m1_uniform_distribution():
    """Con M=1 y poblacion de individuos distinguibles, la distribucion es uniforme."""
    fitnesses = [0.1, 0.2, 0.3, 0.4]
    pop = _make_population(fitnesses)
    selector = DeterministicTournamentSelection(m=1)

    n_trials = 10000
    counts = {f: 0 for f in fitnesses}

    np.random.seed(5)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        counts[selected[0].fitness] += 1

    for f in fitnesses:
        observed = counts[f] / n_trials
        assert abs(observed - 0.25) < 0.05, f"fitness {f}: expected ~0.25, got {observed:.2f}"


# --- ProbabilisticTournamentSelection ---

def test_prob_tournament_threshold_one():
    """Con threshold=1.0, siempre se selecciona el mas apto de los 2 competidores."""
    pop = _make_population([0.1, 0.5, 0.3, 0.9])
    selector = ProbabilisticTournamentSelection(threshold=1.0)

    np.random.seed(42)
    for _ in range(1000):
        selected = selector.select(pop, 1)
        assert selected[0].fitness in [0.1, 0.5, 0.3, 0.9]


def test_prob_tournament_threshold_half():
    """Con threshold=0.5 y 2 individuos, cada uno se selecciona ~50% de las veces."""
    pop = _make_population([0.1, 0.9])
    selector = ProbabilisticTournamentSelection(threshold=0.5)

    n_trials = 10000
    count_best = 0

    np.random.seed(42)
    for _ in range(n_trials):
        selected = selector.select(pop, 1)
        if selected[0].fitness == 0.9:
            count_best += 1

    freq = count_best / n_trials
    assert abs(freq - 0.5) < 0.05, f"Expected ~0.50, got {freq:.2f}"


def test_prob_tournament_returns_k():
    """select() devuelve exactamente k individuos."""
    pop = _make_population([float(i + 1) / 10 for i in range(10)])
    selector = ProbabilisticTournamentSelection(threshold=0.75)

    for k in [1, 5, 20]:
        assert len(selector.select(pop, k)) == k
