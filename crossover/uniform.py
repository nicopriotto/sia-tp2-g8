import numpy as np

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class UniformCrossover(CrossoverOperator):
    def __init__(self, p: float = 0.5):
        self.p = p

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        mask = np.random.random(p1.genes.shape[0]) < self.p

        c1 = p1.genes.copy()
        c2 = p2.genes.copy()
        c1[mask] = p2.genes[mask]
        c2[mask] = p1.genes[mask]

        return (
            Individual(genes=c1, gene_type=p1.gene_type),
            Individual(genes=c2, gene_type=p1.gene_type),
        )
