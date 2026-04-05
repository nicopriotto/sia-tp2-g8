import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class NonUniformMutation(MutationOperator):
    def __init__(self, mutation_rate: float, b: float = 5.0):
        self.mutation_rate = mutation_rate
        self.b = b

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()

        if random.random() < self.mutation_rate:
            if max_generations == 0:
                strength = 1.0
            else:
                strength = (1 - generation / max_generations) ** self.b

            idx = random.randint(0, len(mutated.genes) - 1)
            mutated.genes[idx] = mutated.genes[idx].mutate_delta(strength)

        return mutated
