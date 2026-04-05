import random

from core.individual import Individual
from mutation.mutation_operator import MutationOperator


class GenMutation(MutationOperator):
    """Mutacion que reemplaza exactamente un gen con probabilidad configurable."""

    def __init__(self, mutation_rate: float):
        self.mutation_rate = mutation_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Muta un solo gen con probabilidad mutation_rate."""
        if random.random() < self.mutation_rate:
            index = random.randint(0, len(individual.genes) - 1)
            individual.genes[index] = individual.genes[index].mutate_replace()
        return individual
