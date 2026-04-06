import random

import numpy as np

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class OnePointCrossover(CrossoverOperator):
    """Operador de crossover de un punto."""

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Intercambia genes a partir de un punto de corte aleatorio."""
        n_genes = p1.genes.shape[0]

        cut_point = random.randint(0, n_genes - 1)

        c1_genes = np.vstack([p1.genes[:cut_point], p2.genes[cut_point:]]).copy()
        c2_genes = np.vstack([p2.genes[:cut_point], p1.genes[cut_point:]]).copy()

        child1 = Individual(genes=c1_genes, gene_type=p1.gene_type)
        child2 = Individual(genes=c2_genes, gene_type=p1.gene_type)
        return child1, child2
