import random

from core.individual import Individual
from genes import gene_layout
from mutation.mutation_operator import MutationOperator


class GenMutation(MutationOperator):
    """Mutacion que reemplaza exactamente un gen con probabilidad configurable."""

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Muta un solo gen con probabilidad mutation_rate."""
        if random.random() < self.mutation_rate:
            index = random.randint(0, individual.genes.shape[0] - 1)
            individual.genes[index] = gene_layout.random_genes(individual.gene_type, 1)[0]
            individual.fitness_valid = False
        return individual
