import random

import numpy as np

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class TwoPointCrossover(CrossoverOperator):
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        n_genes = p1.genes.shape[0]

        p1_cut = random.randint(0, n_genes - 1)
        p2_cut = random.randint(0, n_genes - 1)
        if p1_cut > p2_cut:
            p1_cut, p2_cut = p2_cut, p1_cut

        c1_genes = np.vstack([
            p1.genes[:p1_cut],
            p2.genes[p1_cut:p2_cut],
            p1.genes[p2_cut:],
        ]).copy()
        c2_genes = np.vstack([
            p2.genes[:p1_cut],
            p1.genes[p1_cut:p2_cut],
            p2.genes[p2_cut:],
        ]).copy()

        return (
            Individual(genes=c1_genes, gene_type=p1.gene_type),
            Individual(genes=c2_genes, gene_type=p1.gene_type),
        )
