import numpy as np
from core.individual import Individual
from render.cpu_renderer import CPURenderer
from fitness.mse import MSEFitness


def test_individual_random_gene_count():
    ind = Individual.random(10)
    assert len(ind.genes) == 10


def test_individual_fitness_zero_initially():
    ind = Individual.random(5)
    assert ind.fitness == 0.0


def test_individual_copy_independence():
    original = Individual.random(5)
    copied = original.copy()
    original.genes[0].r = 0
    assert copied.genes[0].r != 0 or original.genes[0].r == 0
    # Verificacion mas robusta: modificar y chequear que no se propaga
    original.genes[0].x1 = 0.999
    assert copied.genes[0].x1 != 0.999


def test_compute_fitness_updates_value():
    ind = Individual.random(3)
    target = np.ones((100, 100, 4), dtype=np.float32)
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    ind.compute_fitness(target, renderer, fitness_fn)
    assert ind.fitness > 0.0
