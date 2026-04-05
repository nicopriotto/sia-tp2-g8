from core.individual import Individual
from core.population import Population
from selection.elite import EliteSelection


def _build_population(fitnesses: list[float]) -> Population:
    individuals = [Individual.random(3) for _ in fitnesses]
    for individual, fitness in zip(individuals, fitnesses):
        individual.fitness = fitness
    return Population(individuals=individuals)


def test_elite_k4_slide_example():
    population = _build_population([78, 68, 62, 39, 25, 12, 2])

    selected = EliteSelection().select(population, 4)

    assert [individual.fitness for individual in selected] == [78, 68, 62, 39]


def test_elite_returns_k_individuals():
    population = _build_population([90, 80, 70, 60, 50, 40, 30])
    selector = EliteSelection()

    for k in [1, 3, 5, 7]:
        selected = selector.select(population, k)
        assert len(selected) == k


def test_elite_returns_copies():
    population = _build_population([78, 68, 62, 39, 25, 12, 2])

    selected = EliteSelection().select(population, 4)
    selected[0].fitness = -1
    selected[0].genes[0].r = 0

    assert population.individuals[0].fitness == 78
    assert population.individuals[0].genes[0].r != 0


def test_elite_k_equals_n():
    population = _build_population([10, 50, 20, 40, 30])

    selected = EliteSelection().select(population, 5)

    assert [individual.fitness for individual in selected] == [50, 40, 30, 20, 10]


def test_elite_k_greater_than_n():
    population = _build_population([78, 68, 62, 39, 25, 12, 2])

    selected = EliteSelection().select(population, 12)

    assert [individual.fitness for individual in selected] == [
        78,
        78,
        68,
        68,
        62,
        62,
        39,
        39,
        25,
        25,
        12,
        2,
    ]
