import random
from math import ceil

import numpy as np

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class AnnularCrossover(CrossoverOperator):
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        n_genes = p1.genes.shape[0]
        P = random.randint(0, n_genes - 1)
        L = random.randint(0, ceil(n_genes / 2))

        swap_indices = np.array([(P + i) % n_genes for i in range(L)])

        c1 = p1.genes.copy()
        c2 = p2.genes.copy()
        if len(swap_indices) > 0:
            c1[swap_indices] = p2.genes[swap_indices]
            c2[swap_indices] = p1.genes[swap_indices]

        return (
            Individual(genes=c1, gene_type=p1.gene_type),
            Individual(genes=c2, gene_type=p1.gene_type),
        )
