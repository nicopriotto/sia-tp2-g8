import random

from core.individual import Individual
from crossover.one_point import OnePointCrossover
from genes.triangle_gene import TriangleGene


def _make_gene(value: int) -> TriangleGene:
    return TriangleGene(
        x1=0.01 * value,
        y1=0.02 * value,
        x2=0.03 * value,
        y2=0.04 * value,
        x3=0.05 * value,
        y3=0.06 * value,
        r=value,
        g=value + 1,
        b=value + 2,
        a=min(1.0, 0.1 * value),
    )


def _make_parent(gene_values: list[int], fitness: float = 0.0) -> Individual:
    genes = [_make_gene(value) for value in gene_values]
    return Individual(genes=genes, fitness=fitness)


def test_one_point_child_length():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, child2 = OnePointCrossover().crossover(parent1, parent2)

    assert len(child1.genes) == len(parent1.genes)
    assert len(child2.genes) == len(parent2.genes)


def test_one_point_gene_origin():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])
    seed = 12345

    random.seed(seed)
    cut_point = random.randint(0, len(parent1.genes) - 1)

    random.seed(seed)
    child1, child2 = OnePointCrossover().crossover(parent1, parent2)

    assert child1.genes[:cut_point] == parent1.genes[:cut_point]
    assert child1.genes[cut_point:] == parent2.genes[cut_point:]
    assert child2.genes[:cut_point] == parent2.genes[:cut_point]
    assert child2.genes[cut_point:] == parent1.genes[cut_point:]


def test_one_point_copy_independence():
    parent1 = _make_parent([1, 2, 3, 4])
    parent2 = _make_parent([11, 12, 13, 14])

    child1, _ = OnePointCrossover().crossover(parent1, parent2)
    original_x1 = parent1.genes[0].x1
    original_r = parent2.genes[0].r

    child1.genes[0].x1 = 0.999
    child1.genes[0].r = 255

    assert parent1.genes[0].x1 == original_x1
    assert parent2.genes[0].r == original_r


def test_one_point_fitness_zero():
    parent1 = _make_parent([1, 2, 3], fitness=0.8)
    parent2 = _make_parent([11, 12, 13], fitness=0.6)

    child1, child2 = OnePointCrossover().crossover(parent1, parent2)

    assert child1.fitness == 0.0
    assert child2.fitness == 0.0
