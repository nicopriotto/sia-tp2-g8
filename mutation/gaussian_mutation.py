import numpy as np

from mutation.mutation_operator import MutationOperator
from core.individual import Individual
from genes import gene_layout


class GaussianMutation(MutationOperator):
    """Mutacion gaussiana: perturba genes con distribucion normal.

    sigma       — desviacion para coordenadas geometricas (rango [0, 1])
    sigma_color — desviacion para color RGB/alpha (escala [0, 1], se convierte a [0, 255] internamente)
    """

    def __init__(self, mutation_rate: float, sigma: float = 0.1, sigma_color: float = None,
                 decay_b: float = 0.0):
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.sigma_color = sigma_color if sigma_color is not None else sigma
        self.decay_b = decay_b

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Aplica mutacion gaussiana a los genes del individuo."""
        n_genes = individual.genes.shape[0]
        mask = np.random.random(n_genes) < self.mutation_rate

        if not mask.any():
            return individual.copy()

        # Calcular sigma efectivo con decay temporal
        if self.decay_b > 0 and max_generations > 0:
            progress = generation / max_generations
            decay = (1.0 - progress) ** self.decay_b
            sigma = self.sigma * decay
            sigma_color = self.sigma_color * decay
        else:
            sigma = self.sigma
            sigma_color = self.sigma_color

        mutated = individual.copy()
        n_mutated = int(mask.sum())

        # Geometria (columnas 0-5): sigma en espacio [0, 1]
        noise = np.zeros((n_mutated, gene_layout.N_COLS))
        noise[:, :6] = np.random.normal(0, sigma, size=(n_mutated, 6))
        # RGB (columnas 6-8): sigma_color en [0, 1], convertido a escala [0, 255]
        noise[:, 6:9] = np.random.normal(0, sigma_color * 255, size=(n_mutated, 3))
        # Alpha (columna 9): sigma_color en [0, 1]
        noise[:, 9] = np.random.normal(0, sigma_color, size=n_mutated)
        # active (columna 10) y padding ellipse (columna 5) no se perturban
        if mutated.gene_type == "ellipse":
            noise[:, 5] = 0.0

        mutated.genes[mask] += noise
        gene_layout.clamp(mutated.genes, mutated.gene_type)
        mutated.fitness_valid = False
        return mutated
