import numpy as np

from crossover.crossover_operator import CrossoverOperator
from core.individual import Individual
from genes import gene_layout


class ArithmeticCrossover(CrossoverOperator):
    """Crossover aritmetico: interpola genes entre dos padres."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Aplica crossover aritmetico: interpola cada gen entre ambos padres."""
        alpha = self.alpha

        c1_genes = (alpha * p1.genes + (1 - alpha) * p2.genes).copy()
        c2_genes = (alpha * p2.genes + (1 - alpha) * p1.genes).copy()

        # Preservar active de cada padre
        c1_genes[:, 10] = p1.genes[:, 10]
        c2_genes[:, 10] = p2.genes[:, 10]

        gene_layout.clamp(c1_genes, p1.gene_type)
        gene_layout.clamp(c2_genes, p1.gene_type)

        return (
            Individual(genes=c1_genes, gene_type=p1.gene_type),
            Individual(genes=c2_genes, gene_type=p1.gene_type),
        )
