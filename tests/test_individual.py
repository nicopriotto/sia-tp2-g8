import numpy as np
from core.individual import Individual
from render.cpu_renderer import CPURenderer
from fitness.mse import MSEFitness


def test_individual_random_gene_count():
    ind = Individual.random(10)
    assert ind.genes.shape[0] == 10
    assert ind.genes.shape[1] == 11


def test_individual_fitness_zero_initially():
    ind = Individual.random(5)
    assert ind.fitness == 0.0


def test_individual_copy_independence():
    original = Individual.random(5)
    copied = original.copy()
    original.genes[0, 6] = 0  # r
    assert copied.genes[0, 6] != 0 or original.genes[0, 6] == 0
    original.genes[0, 0] = 0.999  # x1
    assert copied.genes[0, 0] != 0.999


def test_compute_fitness_updates_value():
    ind = Individual.random(3)
    target = np.ones((100, 100, 4), dtype=np.float32)
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    ind.compute_fitness(target, renderer, fitness_fn)
    assert ind.fitness > 0.0
