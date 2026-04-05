from core.individual import Individual
from core.population import Population
from selection.elite import EliteSelection
from survival.additive import AdditiveSurvival


def _make_individual(fitness: float, gene_count: int = 3) -> Individual:
    individual = Individual.random(gene_count)
    individual.fitness = fitness
    return individual


def test_additive_output_size():
    current = Population(individuals=[_make_individual(f) for f in [0.1, 0.2, 0.3, 0.4, 0.5]])
    children = [_make_individual(f) for f in [0.6, 0.7, 0.8, 0.9]]

    new_population = AdditiveSurvival().apply(current, children, EliteSelection())

    assert len(new_population.individuals) == 5


def test_additive_pool_includes_children():
    current = Population(individuals=[_make_individual(f) for f in [0.1, 0.2, 0.3]])
    children = [_make_individual(f) for f in [0.9, 0.8]]

    new_population = AdditiveSurvival().apply(current, children, EliteSelection())
    new_fitnesses = [individual.fitness for individual in new_population.individuals]

    assert 0.9 in new_fitnesses or 0.8 in new_fitnesses


def test_additive_selects_best():
    current = Population(individuals=[_make_individual(f) for f in [0.2, 0.4, 0.1, 0.3]])
    children = [_make_individual(f) for f in [0.6, 0.95, 0.5]]

    new_population = AdditiveSurvival().apply(current, children, EliteSelection())
    new_fitnesses = [individual.fitness for individual in new_population.individuals]

    assert 0.95 in new_fitnesses
