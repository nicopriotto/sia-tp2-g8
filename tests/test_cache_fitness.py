import numpy as np

from core.individual import Individual
from genes import gene_layout
from mutation.gen_mutation import GenMutation


def _make_individual(fitness=0.0, fitness_valid=False):
    genes = gene_layout.random_genes("triangle", 5)
    return Individual(genes=genes, gene_type="triangle", fitness=fitness, fitness_valid=fitness_valid)


def test_compute_fitness_no_reevalua_si_valid():
    class ExplodingRenderer:
        def render(self, genes, width, height, gene_type="triangle"):
            raise RuntimeError("No deberia llamarse render")

    class ExplodingFitness:
        def compute(self, generated, target):
            raise RuntimeError("No deberia llamarse compute")

    ind = _make_individual(fitness=0.5, fitness_valid=True)
    target = np.ones((32, 32, 4), dtype=np.float32)
    ind.compute_fitness(target, ExplodingRenderer(), ExplodingFitness())
    assert ind.fitness == 0.5


def test_copy_preserva_fitness_valid():
    ind = _make_individual(fitness=0.7, fitness_valid=True)
    copia = ind.copy()
    assert copia.fitness_valid is True
    assert copia.fitness == 0.7


def test_mutacion_invalida_cache():
    ind = _make_individual(fitness=0.5, fitness_valid=True)
    mutation = GenMutation(mutation_rate=1.0)
    result = mutation.mutate(ind, generation=1, max_generations=100)
    assert result.fitness_valid is False


def test_mutacion_no_aplica_preserva_cache():
    ind = _make_individual(fitness=0.5, fitness_valid=True)
    mutation = GenMutation(mutation_rate=0.0)
    result = mutation.mutate(ind, generation=1, max_generations=100)
    assert result.fitness_valid is True


def test_to_dict_no_incluye_fitness_valid():
    ind = _make_individual(fitness=0.5, fitness_valid=True)
    d = ind.to_dict()
    assert "fitness_valid" not in d
