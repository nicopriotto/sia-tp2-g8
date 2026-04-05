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


from survival.exclusive import ExclusiveSurvival


def test_exclusive_k_greater_n_no_parents():
    parent_fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5]
    child_fitnesses = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    current = Population(individuals=[_make_individual(f) for f in parent_fitnesses])
    children = [_make_individual(f) for f in child_fitnesses]

    new_pop = ExclusiveSurvival().apply(current, children, EliteSelection())

    assert len(new_pop.individuals) == 5
    new_fitnesses = [ind.fitness for ind in new_pop.individuals]
    for f in new_fitnesses:
        assert f not in parent_fitnesses, f"Parent fitness {f} found in new population"


def test_exclusive_k_less_n_all_children():
    current = Population(individuals=[_make_individual(f) for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]])
    child_fitnesses = [0.11, 0.22, 0.33]
    children = [_make_individual(f) for f in child_fitnesses]

    new_pop = ExclusiveSurvival().apply(current, children, EliteSelection())

    assert len(new_pop.individuals) == 10
    new_fitnesses = [ind.fitness for ind in new_pop.individuals]
    for f in child_fitnesses:
        assert f in new_fitnesses, f"Child fitness {f} missing from new population"


def test_exclusive_output_size():
    selector = EliteSelection()
    for n, k in [(5, 8), (5, 3), (5, 5), (10, 1), (3, 10)]:
        current = Population(individuals=[_make_individual(float(i) / n) for i in range(n)])
        children = [_make_individual(float(i) / (k + 1)) for i in range(k)]
        new_pop = ExclusiveSurvival().apply(current, children, selector)
        assert len(new_pop.individuals) == n, f"N={n}, K={k}: expected {n}, got {len(new_pop.individuals)}"


def test_exclusive_k_equals_n():
    child_fitnesses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    current = Population(individuals=[_make_individual(f + 0.05) for f in child_fitnesses])
    children = [_make_individual(f) for f in child_fitnesses]

    new_pop = ExclusiveSurvival().apply(current, children, EliteSelection())

    assert len(new_pop.individuals) == 6
    new_fitnesses = sorted(ind.fitness for ind in new_pop.individuals)
    assert new_fitnesses == sorted(child_fitnesses)
