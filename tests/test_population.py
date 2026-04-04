import pytest
import numpy as np
from core.individual import Individual
from core.population import Population
from genes.triangle_gene import TriangleGene


def test_population_size_after_random():
    pop = Population.random(15, 5)
    assert len(pop.individuals) == 15


def test_population_best_is_highest():
    individuals = [Individual.random(3) for _ in range(3)]
    individuals[0].fitness = 0.3
    individuals[1].fitness = 0.8
    individuals[2].fitness = 0.5
    pop = Population(individuals=individuals)
    assert pop.best.fitness == 0.8


def test_fitness_std_zero_equal_fitness():
    individuals = [Individual.random(3) for _ in range(5)]
    for ind in individuals:
        ind.fitness = 0.5
    pop = Population(individuals=individuals)
    assert pop.fitness_std == pytest.approx(0.0)
