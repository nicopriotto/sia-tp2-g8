import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class MultiGenMutation(MutationOperator):
    def __init__(self, mutation_rate: float, max_genes: int):
        self.mutation_rate = mutation_rate
        self.max_genes = max_genes

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()

        if random.random() < self.mutation_rate:
            n_genes = len(mutated.genes)
            m = min(self.max_genes, n_genes)
            k = random.randint(1, m)
            indices = random.sample(range(n_genes), k)
            for i in indices:
                mutated.genes[i] = mutated.genes[i].mutate_replace()

        return mutated
