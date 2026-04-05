import random

from core.individual import Individual
from genes.triangle_gene import TriangleGene
from mutation.gen_mutation import GenMutation


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
        a=min(1.0, 0.01 * value),
    )


def _make_individual(size: int = 10) -> Individual:
    return Individual(genes=[_make_gene(index + 1) for index in range(size)])


def _gene_signature(gene: TriangleGene) -> tuple[float, float, float, float, float, float, int, int, int, float]:
    return (
        gene.x1,
        gene.y1,
        gene.x2,
        gene.y2,
        gene.x3,
        gene.y3,
        gene.r,
        gene.g,
        gene.b,
        gene.a,
    )


def _changed_gene_count(before: Individual, after: Individual) -> int:
    return sum(
        _gene_signature(before_gene) != _gene_signature(after_gene)
        for before_gene, after_gene in zip(before.genes, after.genes)
    )


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
    mutation = UniformMutation(mutation_rate=0.0)
    original = _make_individual(20)

    for _ in range(100):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        assert _changed_gene_count(original, mutated) == 0


def test_uniform_mutation_rate_one():
    random.seed(2)
    mutation = UniformMutation(mutation_rate=1.0)
    original = _make_individual(20)

    mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
    assert _changed_gene_count(original, mutated) == 20


def test_uniform_mutation_independent():
    random.seed(3)
    mutation = UniformMutation(mutation_rate=0.5)
    original = _make_individual(10)

    per_gene_count = [0] * 10
    n_trials = 1000

    for _ in range(n_trials):
        mutated = mutation.mutate(original.copy(), generation=0, max_generations=100)
        for i in range(10):
            if _gene_signature(original.genes[i]) != _gene_signature(mutated.genes[i]):
                per_gene_count[i] += 1

    for i, count in enumerate(per_gene_count):
        freq = count / n_trials
        assert 0.35 <= freq <= 0.65, f"Gene {i}: expected ~0.50, got {freq:.2f}"


def test_uniform_mutation_approx_rate():
    random.seed(4)
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
