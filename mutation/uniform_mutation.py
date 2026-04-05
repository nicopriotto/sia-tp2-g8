import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class UniformMutation(MutationOperator):
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()
        for i in range(len(mutated.genes)):
            if random.random() < self.mutation_rate:
                mutated.genes[i] = mutated.genes[i].mutate_replace()
        return mutated
