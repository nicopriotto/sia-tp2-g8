import numpy as np

from core.individual import Individual
from genes import gene_layout
from mutation.mutation_operator import MutationOperator


class UniformMutation(MutationOperator):
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()
        n_genes = mutated.genes.shape[0]
        mask = np.random.random(n_genes) < self.mutation_rate

        if mask.any():
            k = int(mask.sum())
            mutated.genes[mask] = gene_layout.random_genes(mutated.gene_type, k)
            mutated.fitness_valid = False

        return mutated
