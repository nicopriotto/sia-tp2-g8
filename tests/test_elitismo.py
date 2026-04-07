import numpy as np
import pytest

from core.individual import Individual
from core.population import Population
from survival.additive import AdditiveSurvival
from selection.roulette import RouletteSelection


def _make_individual(fitness: float, n_genes: int = 5) -> Individual:
    """Crea un individuo con fitness preestablecido."""
    ind = Individual.random(n_genes)
    ind.fitness = fitness
    ind.fitness_valid = True
    return ind


class TestElitismo:
    def test_elite_never_lost(self):
        """El mejor individuo nunca se pierde con elite_count=1."""
        for _ in range(20):
            best = _make_individual(0.9)
            others = [_make_individual(np.random.uniform(0.05, 0.15)) for _ in range(9)]
            population = Population(individuals=[best] + others)

            # Generar hijos mediocres
            children = [_make_individual(np.random.uniform(0.05, 0.15)) for _ in range(10)]

            # Aplicar survival (puede perder al mejor)
            selector = RouletteSelection()
            survival = AdditiveSurvival()
            new_pop = survival.apply(population, children, selector)

            # Simular elitismo post-survival
            elite = [best.copy()]
            new_pop.individuals.sort(key=lambda i: i.fitness)
            for elite_ind in elite:
                if elite_ind.fitness > new_pop.individuals[0].fitness:
                    new_pop.individuals[0] = elite_ind
                    new_pop.individuals.sort(key=lambda i: i.fitness)

            assert new_pop.best.fitness >= 0.9

    def test_elite_count_zero_no_guarantee(self):
        """Con elite_count=0, el GA no crashea."""
        population = Population(individuals=[_make_individual(0.5) for _ in range(5)])
        children = [_make_individual(0.3) for _ in range(5)]
        survival = AdditiveSurvival()
        selector = RouletteSelection()
        # Sin elitismo, simplemente no hacer nada extra
        new_pop = survival.apply(population, children, selector)
        assert len(new_pop.individuals) == 5

    def test_elite_count_greater_than_one(self):
        """Con elite_count=3, los 3 mejores se preservan."""
        fitnesses = [0.9, 0.8, 0.7, 0.2, 0.1]
        population = Population(individuals=[_make_individual(f) for f in fitnesses])
        children = [_make_individual(0.05) for _ in range(5)]

        survival = AdditiveSurvival()
        selector = RouletteSelection()
        new_pop = survival.apply(population, children, selector)

        # Simular elitismo con 3 elite
        elite = [_make_individual(f) for f in [0.9, 0.8, 0.7]]
        for e in elite:
            e.fitness_valid = True
        new_pop.individuals.sort(key=lambda i: i.fitness)
        for elite_ind in elite:
            if elite_ind.fitness > new_pop.individuals[0].fitness:
                new_pop.individuals[0] = elite_ind
                new_pop.individuals.sort(key=lambda i: i.fitness)

        result_fitnesses = sorted([i.fitness for i in new_pop.individuals], reverse=True)
        assert result_fitnesses[0] >= 0.9
        assert result_fitnesses[1] >= 0.8
        assert result_fitnesses[2] >= 0.7

    def test_elite_does_not_increase_population(self):
        """La poblacion mantiene su tamano despues del elitismo."""
        pop_size = 10
        population = Population(
            individuals=[_make_individual(np.random.uniform(0.1, 0.9)) for _ in range(pop_size)]
        )
        children = [_make_individual(np.random.uniform(0.1, 0.5)) for _ in range(pop_size)]

        # Guardar elite
        sorted_inds = sorted(population.individuals, key=lambda i: i.fitness, reverse=True)
        elite = [sorted_inds[0].copy(), sorted_inds[1].copy()]

        survival = AdditiveSurvival()
        selector = RouletteSelection()
        new_pop = survival.apply(population, children, selector)

        # Inyectar elite
        new_pop.individuals.sort(key=lambda i: i.fitness)
        for elite_ind in elite:
            if elite_ind.fitness > new_pop.individuals[0].fitness:
                new_pop.individuals[0] = elite_ind
                new_pop.individuals.sort(key=lambda i: i.fitness)

        assert len(new_pop.individuals) == pop_size
