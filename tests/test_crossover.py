import random
from unittest.mock import patch

from core.individual import Individual
from crossover.annular import AnnularCrossover
from crossover.one_point import OnePointCrossover
from crossover.two_point import TwoPointCrossover
from crossover.uniform import UniformCrossover
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


def test_two_point_child_length():
    parent1 = _make_parent(list(range(1, 11)))
    parent2 = _make_parent(list(range(11, 21)))

    child1, child2 = TwoPointCrossover().crossover(parent1, parent2)

    assert len(child1.genes) == 10
    assert len(child2.genes) == 10


def test_two_point_segment_swapped():
    parent1 = _make_parent([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parent2 = _make_parent([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    seed = 42

    random.seed(seed)
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    p1, p2 = (a, b) if a <= b else (b, a)

    random.seed(seed)
    child1, child2 = TwoPointCrossover().crossover(parent1, parent2)

    # Outside segment: child1 gets p1's genes, child2 gets p2's genes
    for i in list(range(0, p1)) + list(range(p2, 10)):
        assert child1.genes[i].r == parent1.genes[i].r
        assert child2.genes[i].r == parent2.genes[i].r
    # Inside segment [p1:p2]: swapped
    for i in range(p1, p2):
        assert child1.genes[i].r == parent2.genes[i].r
        assert child2.genes[i].r == parent1.genes[i].r


def test_two_point_copy_independence():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, _ = TwoPointCrossover().crossover(parent1, parent2)
    original_r_p1 = parent1.genes[0].r
    original_r_p2 = parent2.genes[0].r

    child1.genes[0].r = 999

    assert parent1.genes[0].r == original_r_p1
    assert parent2.genes[0].r == original_r_p2


def test_uniform_p0_no_swap():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, child2 = UniformCrossover(p=0.0).crossover(parent1, parent2)

    for i in range(5):
        assert child1.genes[i].r == parent1.genes[i].r
        assert child2.genes[i].r == parent2.genes[i].r


def test_uniform_p1_full_swap():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, child2 = UniformCrossover(p=1.0).crossover(parent1, parent2)

    for i in range(5):
        assert child1.genes[i].r == parent2.genes[i].r
        assert child2.genes[i].r == parent1.genes[i].r


def test_uniform_p05_approx_half():
    parent1 = _make_parent(list(range(1, 21)))
    parent2 = _make_parent(list(range(21, 41)))
    op = UniformCrossover(p=0.5)

    total_swapped = 0
    n_trials = 1000

    random.seed(99)
    for _ in range(n_trials):
        child1, _ = op.crossover(parent1, parent2)
        swapped = sum(1 for i in range(20) if child1.genes[i].r == parent2.genes[i].r)
        total_swapped += swapped

    avg_fraction = total_swapped / (n_trials * 20)
    assert 0.40 <= avg_fraction <= 0.60, f"Expected ~0.50, got {avg_fraction:.2f}"


def test_uniform_copy_independence():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, _ = UniformCrossover(p=0.5).crossover(parent1, parent2)
    original_r_p1 = parent1.genes[0].r
    original_r_p2 = parent2.genes[0].r

    child1.genes[0].r = 999

    assert parent1.genes[0].r == original_r_p1
    assert parent2.genes[0].r == original_r_p2


def test_annular_slide_example():
    """P=11, L=5 on 12-gene chromosome swaps indices [11,0,1,2,3]."""
    parent1 = _make_parent(list(range(1, 13)))
    parent2 = _make_parent(list(range(101, 113)))

    with patch("crossover.annular.random.randint", side_effect=[11, 5]):
        child1, child2 = AnnularCrossover().crossover(parent1, parent2)

    swap_indices = {11, 0, 1, 2, 3}
    for i in range(12):
        if i in swap_indices:
            assert child1.genes[i].r == parent2.genes[i].r
            assert child2.genes[i].r == parent1.genes[i].r
        else:
            assert child1.genes[i].r == parent1.genes[i].r
            assert child2.genes[i].r == parent2.genes[i].r


def test_annular_wrapping():
    """n=10, P=8, L=4 => swap indices [8,9,0,1]."""
    parent1 = _make_parent(list(range(1, 11)))
    parent2 = _make_parent(list(range(101, 111)))

    with patch("crossover.annular.random.randint", side_effect=[8, 4]):
        child1, child2 = AnnularCrossover().crossover(parent1, parent2)

    swap_indices = {8, 9, 0, 1}
    for i in range(10):
        if i in swap_indices:
            assert child1.genes[i].r == parent2.genes[i].r
        else:
            assert child1.genes[i].r == parent1.genes[i].r


def test_annular_l0_no_swap():
    """L=0 means no genes are swapped — children are copies of parents."""
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    with patch("crossover.annular.random.randint", side_effect=[2, 0]):
        child1, child2 = AnnularCrossover().crossover(parent1, parent2)

    for i in range(5):
        assert child1.genes[i].r == parent1.genes[i].r
        assert child2.genes[i].r == parent2.genes[i].r


def test_annular_copy_independence():
    parent1 = _make_parent([1, 2, 3, 4, 5])
    parent2 = _make_parent([11, 12, 13, 14, 15])

    child1, _ = AnnularCrossover().crossover(parent1, parent2)
    original_r_p1 = parent1.genes[0].r
    original_r_p2 = parent2.genes[0].r

    child1.genes[0].r = 999

    assert parent1.genes[0].r == original_r_p1
    assert parent2.genes[0].r == original_r_p2
