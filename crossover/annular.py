import random
from math import ceil

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class AnnularCrossover(CrossoverOperator):
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        n_genes = len(p1.genes)
        P = random.randint(0, n_genes - 1)
        L = random.randint(0, ceil(n_genes / 2))

        swap_indices = set((P + i) % n_genes for i in range(L))

        genes_h1 = []
        genes_h2 = []
        for i in range(n_genes):
            if i in swap_indices:
                genes_h1.append(p2.genes[i].copy())
                genes_h2.append(p1.genes[i].copy())
            else:
                genes_h1.append(p1.genes[i].copy())
                genes_h2.append(p2.genes[i].copy())

        return Individual(genes=genes_h1, fitness=0.0), Individual(genes=genes_h2, fitness=0.0)
