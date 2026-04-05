import random

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class TwoPointCrossover(CrossoverOperator):
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        n_genes = len(p1.genes)

        p1_cut = random.randint(0, n_genes - 1)
        p2_cut = random.randint(0, n_genes - 1)
        if p1_cut > p2_cut:
            p1_cut, p2_cut = p2_cut, p1_cut

        genes_h1 = (
            [g.copy() for g in p1.genes[:p1_cut]]
            + [g.copy() for g in p2.genes[p1_cut:p2_cut]]
            + [g.copy() for g in p1.genes[p2_cut:]]
        )
        genes_h2 = (
            [g.copy() for g in p2.genes[:p1_cut]]
            + [g.copy() for g in p1.genes[p1_cut:p2_cut]]
            + [g.copy() for g in p2.genes[p2_cut:]]
        )

        return Individual(genes=genes_h1, fitness=0.0), Individual(genes=genes_h2, fitness=0.0)
