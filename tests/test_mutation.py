import random

import numpy as np

from core.individual import Individual
from genes.triangle_gene import TriangleGene
from mutation.gen_mutation import GenMutation


def _make_gene_row(value: int) -> np.ndarray:
    return np.array([
        0.01 * value, 0.02 * value, 0.03 * value, 0.04 * value,
        0.05 * value, 0.06 * value,
        value, value + 1, value + 2,
        min(1.0, 0.01 * value),
        1.0,
    ], dtype=np.float64)


def _make_individual(size: int = 10) -> Individual:
    genes = np.array([_make_gene_row(i + 1) for i in range(size)])
    return Individual(genes=genes, gene_type="triangle")


def _gene_signature(row: np.ndarray) -> tuple:
    return tuple(row[:10])


def _changed_gene_count(before: Individual, after: Individual) -> int:
    return int(np.sum(~np.all(np.isclose(before.genes[:, :10], after.genes[:, :10]), axis=1)))


def test_gen_mutation_rate_zero():
    random.seed(1234)
    mutation = GenMutation(mutation_rate=0.0)
    original = _make_individual()

    for _ in range(100):
        candidate = original.copy()
        mutated = mutation.mutate(candidate, generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 0


def test_gen_mutation_rate_one():
    random.seed(4321)
    mutation = GenMutation(mutation_rate=1.0)
    original = _make_individual()

    mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)

    assert _changed_gene_count(original, mutated) == 1


def test_gen_mutation_exactly_one_gene():
    random.seed(9999)
    mutation = GenMutation(mutation_rate=1.0)
    original = _make_individual()

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 1


def test_gen_mutation_approximate_rate():
    random.seed(2024)
    mutation = GenMutation(mutation_rate=0.5)
    original = _make_individual()
    mutated_count = 0

    for _ in range(1000):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        if _changed_gene_count(original, mutated) > 0:
            mutated_count += 1

    assert 400 <= mutated_count <= 600


from mutation.multigen_mutation import MultiGenMutation


def test_multigen_m1_equals_gen():
    random.seed(42)
    mutation = MultiGenMutation(mutation_rate=1.0, max_genes=1)
    original = _make_individual(10)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 1


def test_multigen_max_genes_respected():
    random.seed(7)
    mutation = MultiGenMutation(mutation_rate=1.0, max_genes=3)
    original = _make_individual(10)

    for _ in range(1000):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) <= 3


def test_multigen_rate_zero():
    random.seed(99)
    mutation = MultiGenMutation(mutation_rate=0.0, max_genes=5)
    original = _make_individual(10)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 0


def test_multigen_mutates_at_least_one():
    random.seed(11)
    mutation = MultiGenMutation(mutation_rate=1.0, max_genes=5)
    original = _make_individual(10)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) >= 1


from mutation.uniform_mutation import UniformMutation


def test_uniform_mutation_rate_zero():
    random.seed(1)
    np.random.seed(1)
    mutation = UniformMutation(mutation_rate=0.0)
    original = _make_individual(20)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 0


def test_uniform_mutation_rate_one():
    random.seed(2)
    np.random.seed(2)
    mutation = UniformMutation(mutation_rate=1.0)
    original = _make_individual(20)

    mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
    assert _changed_gene_count(original, mutated) == 20


def test_uniform_mutation_independent():
    random.seed(3)
    np.random.seed(3)
    mutation = UniformMutation(mutation_rate=0.5)
    original = _make_individual(10)

    per_gene_count = [0] * 10
    n_trials = 1000

    for _ in range(n_trials):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        for i in range(10):
            if not np.allclose(original.genes[i, :10], mutated.genes[i, :10]):
                per_gene_count[i] += 1

    for i, count in enumerate(per_gene_count):
        freq = count / n_trials
        assert 0.35 <= freq <= 0.65, f"Gene {i}: expected ~0.50, got {freq:.2f}"


def test_uniform_mutation_approx_rate():
    random.seed(4)
    np.random.seed(4)
    mutation = UniformMutation(mutation_rate=0.1)
    original = _make_individual(50)

    total_changed = 0
    n_trials = 1000

    for _ in range(n_trials):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        total_changed += _changed_gene_count(original, mutated)

    avg = total_changed / n_trials
    assert 3 <= avg <= 7, f"Expected avg ~5, got {avg:.2f}"


from mutation.complete_mutation import CompleteMutation


def test_complete_mutation_rate_zero():
    random.seed(5)
    mutation = CompleteMutation(mutation_rate=0.0)
    original = _make_individual(10)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 0


def test_complete_mutation_rate_one():
    random.seed(6)
    mutation = CompleteMutation(mutation_rate=1.0)
    original = _make_individual(10)

    mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
    assert _changed_gene_count(original, mutated) == 10


def test_complete_mutation_all_or_nothing():
    random.seed(7)
    mutation = CompleteMutation(mutation_rate=0.5)
    original = _make_individual(10)

    for _ in range(200):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        changed = _changed_gene_count(original, mutated)
        assert changed == 0 or changed == 10, f"Expected 0 or 10, got {changed}"


def test_complete_mutation_approx_rate():
    random.seed(8)
    mutation = CompleteMutation(mutation_rate=0.5)
    original = _make_individual(10)

    full_mutations = 0
    for _ in range(1000):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        if _changed_gene_count(original, mutated) == 10:
            full_mutations += 1

    assert 400 <= full_mutations <= 600, f"Expected ~500, got {full_mutations}"


from mutation.non_uniform_mutation import NonUniformMutation


def _row_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """Promedio de diferencias absolutas en coordenadas y alpha."""
    return float(
        abs(r1[0] - r2[0]) + abs(r1[1] - r2[1]) +
        abs(r1[2] - r2[2]) + abs(r1[3] - r2[3]) +
        abs(r1[4] - r2[4]) + abs(r1[5] - r2[5]) +
        abs(r1[9] - r2[9])
    ) / 7


def test_non_uniform_strength_decreases():
    mutation = NonUniformMutation(mutation_rate=1.0, b=5.0)
    max_gen = 100

    s0 = (1 - 0 / max_gen) ** 5.0
    s50 = (1 - 50 / max_gen) ** 5.0
    s100 = (1 - 100 / max_gen) ** 5.0

    assert s0 > s50 > s100
    assert s0 == 1.0
    assert abs(s50 - 0.03125) < 1e-9
    assert s100 == 0.0


def test_non_uniform_early_large_delta():
    random.seed(10)
    np.random.seed(10)
    mutation = NonUniformMutation(mutation_rate=1.0, b=5.0)
    original = _make_individual(10)

    early_distances = []
    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=1000)
        diffs = ~np.all(np.isclose(original.genes[:, :10], mutated.genes[:, :10]), axis=1)
        changed = np.where(diffs)[0]
        if len(changed) > 0:
            early_distances.append(_row_distance(original.genes[changed[0]], mutated.genes[changed[0]]))

    late_distances = []
    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=900, max_generations=1000)
        diffs = ~np.all(np.isclose(original.genes[:, :10], mutated.genes[:, :10]), axis=1)
        changed = np.where(diffs)[0]
        if len(changed) > 0:
            late_distances.append(_row_distance(original.genes[changed[0]], mutated.genes[changed[0]]))

    avg_early = sum(early_distances) / len(early_distances) if early_distances else 0
    avg_late = sum(late_distances) / len(late_distances) if late_distances else 0
    assert avg_early > avg_late


def test_non_uniform_late_small_delta():
    random.seed(20)
    np.random.seed(20)
    mutation = NonUniformMutation(mutation_rate=1.0, b=5.0)
    original = _make_individual(10)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=1000, max_generations=1000)
        for i in range(10):
            o = original.genes[i]
            m = mutated.genes[i]
            assert abs(o[0] - m[0]) < 0.01  # x1
            assert abs(o[1] - m[1]) < 0.01  # y1
            assert abs(o[2] - m[2]) < 0.01  # x2
            assert abs(o[3] - m[3]) < 0.01  # y2
            assert abs(o[4] - m[4]) < 0.01  # x3
            assert abs(o[5] - m[5]) < 0.01  # y3
            assert abs(o[6] - m[6]) < 3     # r
            assert abs(o[7] - m[7]) < 3     # g
            assert abs(o[8] - m[8]) < 3     # b


def test_non_uniform_never_degenerate():
    random.seed(30)
    np.random.seed(30)
    mutation = NonUniformMutation(mutation_rate=1.0, b=5.0)
    original = _make_individual(10)

    for t in [0, 500, 1000]:
        for _ in range(333):
            mutated = mutation.mutate(original.copy(), generation=t, max_generations=1000)
            for row in mutated.genes:
                assert 0.0 <= row[0] <= 1.0
                assert 0.0 <= row[1] <= 1.0
                assert 0.0 <= row[2] <= 1.0
                assert 0.0 <= row[3] <= 1.0
                assert 0.0 <= row[4] <= 1.0
                assert 0.0 <= row[5] <= 1.0
                assert 0 <= row[6] <= 255
                assert 0 <= row[7] <= 255
                assert 0 <= row[8] <= 255
                assert 0.0 <= row[9] <= 1.0
