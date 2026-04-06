import numpy as np

from mutation.mutation_operator import MutationOperator
from core.individual import Individual
from genes import gene_layout


class GaussianMutation(MutationOperator):
    """Mutacion gaussiana: perturba genes con distribucion normal."""

    def __init__(self, mutation_rate: float, sigma: float = 0.1):
        self.mutation_rate = mutation_rate
        self.sigma = sigma

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Aplica mutacion gaussiana a los genes del individuo."""
        n_genes = individual.genes.shape[0]
        mask = np.random.random(n_genes) < self.mutation_rate

        if not mask.any():
            result = individual.copy()
            return result

        mutated = individual.copy()
        n_mutated = int(mask.sum())
        noise = np.random.normal(0, self.sigma, size=(n_mutated, gene_layout.N_COLS))
        noise[:, 6:9] *= 255  # scale RGB noise
        noise[:, 10] = 0  # don't perturb active
        if mutated.gene_type == "ellipse":
            noise[:, 5] = 0  # don't perturb padding
        mutated.genes[mask] += noise
        gene_layout.clamp(mutated.genes, mutated.gene_type)
        mutated.fitness_valid = False
        return mutated
