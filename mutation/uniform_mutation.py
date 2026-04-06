import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class UniformMutation(MutationOperator):
    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()
        changed = False
        for i in range(len(mutated.genes)):
            if random.random() < self.mutation_rate:
                mutated.genes[i] = mutated.genes[i].mutate_replace()
                changed = True
        if changed:
            mutated.fitness_valid = False
        return mutated
