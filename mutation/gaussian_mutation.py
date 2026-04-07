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
                 decay_b: float = 0.0, swap_rate: float = 0.0):
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.sigma_color = sigma_color if sigma_color is not None else sigma
        self.decay_b = decay_b
        self.swap_rate = swap_rate

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Aplica mutacion gaussiana per-float a los genes del individuo."""
        n_genes = individual.genes.shape[0]

        # Mascara per-float: shape (n_genes, N_COLS)
        mask = np.random.random((n_genes, gene_layout.N_COLS)) < self.mutation_rate

        # Nunca mutar columna 10 (active)
        mask[:, 10] = False
        # Nunca mutar columna 5 (padding) para ellipses
        if individual.gene_type == "ellipse":
            mask[:, 5] = False

        mutated = individual.copy()
        changed = False

        if mask.any():
            # Calcular sigma efectivo con decay temporal
            if self.decay_b > 0 and max_generations > 0:
                progress = generation / max_generations
                decay = (1.0 - progress) ** self.decay_b
                sigma = self.sigma * decay
                sigma_color = self.sigma_color * decay
            else:
                sigma = self.sigma
                sigma_color = self.sigma_color

            # Generar ruido con escala apropiada por columna
            noise = np.zeros_like(mutated.genes)
            noise[:, :6] = np.random.normal(0, sigma, size=(n_genes, 6))
            noise[:, 6:9] = np.random.normal(0, sigma_color * 255, size=(n_genes, 3))
            noise[:, 9] = np.random.normal(0, sigma_color, size=n_genes)

            # Aplicar ruido solo donde la mascara es True
            mutated.genes[mask] += noise[mask]
            gene_layout.clamp(mutated.genes, mutated.gene_type)
            changed = True

        # Swap de Z-index (independiente de la perturbacion)
        if self.swap_rate > 0 and n_genes >= 2 and np.random.random() < self.swap_rate:
            i, j = np.random.choice(n_genes, size=2, replace=False)
            mutated.genes[[i, j]] = mutated.genes[[j, i]]
            changed = True

        if changed:
            mutated.fitness_valid = False
        return mutated
