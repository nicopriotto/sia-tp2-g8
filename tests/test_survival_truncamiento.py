import pytest

from core.individual import Individual
from core.population import Population
from survival.additive import AdditiveSurvival
from selection.roulette import RouletteSelection


def _make_individual(fitness: float) -> Individual:
    ind = Individual.random(5)
    ind.fitness = fitness
    ind.fitness_valid = True
    return ind


class TestAdditiveTruncamiento:
    def test_additive_keeps_best(self):
        """Sobreviven los N con mayor fitness del pool combinado."""
        population = Population(
            individuals=[_make_individual(f) for f in [0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        children = [_make_individual(f) for f in [0.15, 0.35, 0.6]]

        survival = AdditiveSurvival()
        result = survival.apply(population, children, RouletteSelection())

        result_fitnesses = sorted([i.fitness for i in result.individuals], reverse=True)
        expected = [0.6, 0.5, 0.4, 0.35, 0.3]
        assert result_fitnesses == pytest.approx(expected)

    def test_additive_preserves_population_size(self):
        """La poblacion mantiene su tamano original."""
        population = Population(individuals=[_make_individual(0.5) for _ in range(10)])
        children = [_make_individual(0.3) for _ in range(10)]

        result = AdditiveSurvival().apply(population, children, RouletteSelection())
        assert len(result.individuals) == 10

    def test_additive_deterministic(self):
        """El resultado es identico en multiples ejecuciones."""
        population = Population(
            individuals=[_make_individual(f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
        )
        children = [_make_individual(f) for f in [0.2, 0.4, 0.6]]

        survival = AdditiveSurvival()
        results = []
        for _ in range(10):
            r = survival.apply(population, children, RouletteSelection())
            results.append(sorted([i.fitness for i in r.individuals], reverse=True))

        for r in results[1:]:
            assert r == pytest.approx(results[0])

    def test_additive_all_children_worse(self):
        """Si todos los hijos son peores, solo sobreviven los padres."""
        population = Population(
            individuals=[_make_individual(f) for f in [0.6, 0.7, 0.8, 0.9]]
        )
        children = [_make_individual(f) for f in [0.1, 0.2, 0.3]]

        result = AdditiveSurvival().apply(population, children, RouletteSelection())
        result_fitnesses = sorted([i.fitness for i in result.individuals], reverse=True)
        assert result_fitnesses == pytest.approx([0.9, 0.8, 0.7, 0.6])

    def test_additive_all_children_better(self):
        """Si todos los hijos son mejores, solo sobreviven los hijos."""
        population = Population(
            individuals=[_make_individual(f) for f in [0.1, 0.2, 0.3]]
        )
        children = [_make_individual(f) for f in [0.7, 0.8, 0.9]]

        result = AdditiveSurvival().apply(population, children, RouletteSelection())
        result_fitnesses = sorted([i.fitness for i in result.individuals], reverse=True)
        assert result_fitnesses == pytest.approx([0.9, 0.8, 0.7])
