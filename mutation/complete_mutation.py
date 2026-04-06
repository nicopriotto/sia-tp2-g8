import random

from core.individual import Individual
from genes import gene_layout
from mutation.mutation_operator import MutationOperator


class CompleteMutation(MutationOperator):
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()
        if random.random() < self.mutation_rate:
            n_genes = mutated.genes.shape[0]
            mutated.genes = gene_layout.random_genes(mutated.gene_type, n_genes)
            mutated.fitness_valid = False
        return mutated
