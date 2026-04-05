import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class CompleteMutation(MutationOperator):
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()
        if random.random() < self.mutation_rate:
            for i in range(len(mutated.genes)):
                mutated.genes[i] = mutated.genes[i].mutate_replace()
        return mutated
