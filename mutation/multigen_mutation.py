import random

import numpy as np

from core.individual import Individual
from genes import gene_layout
from mutation.mutation_operator import MutationOperator


class MultiGenMutation(MutationOperator):
    def __init__(self, mutation_rate: float, max_genes: int):
        self.mutation_rate = mutation_rate
        self.max_genes = max_genes

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()

        if random.random() < self.mutation_rate:
            n_genes = mutated.genes.shape[0]
            m = min(self.max_genes, n_genes)
            k = random.randint(1, m)
            indices = random.sample(range(n_genes), k)
            new_rows = gene_layout.random_genes(mutated.gene_type, k)
            for i, idx in enumerate(indices):
                mutated.genes[idx] = new_rows[i]
            mutated.fitness_valid = False

        return mutated
