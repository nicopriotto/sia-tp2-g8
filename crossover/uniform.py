import random

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class UniformCrossover(CrossoverOperator):
    def __init__(self, p: float = 0.5):
        self.p = p

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        genes_h1 = []
        genes_h2 = []

        for g1, g2 in zip(p1.genes, p2.genes):
            if random.random() < self.p:
                genes_h1.append(g2.copy())
                genes_h2.append(g1.copy())
            else:
                genes_h1.append(g1.copy())
                genes_h2.append(g2.copy())

        return Individual(genes=genes_h1, fitness=0.0), Individual(genes=genes_h2, fitness=0.0)
