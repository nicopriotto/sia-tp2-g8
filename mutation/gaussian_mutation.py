import random

from mutation.mutation_operator import MutationOperator
from core.individual import Individual


class GaussianMutation(MutationOperator):
    """Mutacion gaussiana: perturba genes con distribucion normal."""

    def __init__(self, mutation_rate: float, sigma: float = 0.1):
        """
        Args:
            mutation_rate: Probabilidad de mutar cada gen.
            sigma: Desviacion estandar de la distribucion normal para la perturbacion.
        """
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Aplica mutacion gaussiana a los genes del individuo.

        Por cada gen, con probabilidad mutation_rate, aplica mutate_gaussian(sigma).
        """
        new_genes = []
        for gene in individual.genes:
            if random.random() < self.mutation_rate:
                new_genes.append(gene.mutate_gaussian(self.sigma))
            else:
                new_genes.append(gene.copy())
        return Individual(genes=new_genes)
