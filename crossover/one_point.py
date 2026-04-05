import random

from core.individual import Individual
from crossover.crossover_operator import CrossoverOperator


class OnePointCrossover(CrossoverOperator):
    """Operador de crossover de un punto."""

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Intercambia genes a partir de un punto de corte aleatorio."""
        n_genes = len(p1.genes)
        if len(p2.genes) != n_genes:
            raise ValueError("Ambos padres deben tener la misma cantidad de genes.")

        cut_point = random.randint(0, n_genes - 1)

        child1_genes = [gene.copy() for gene in p1.genes[:cut_point]]
        child1_genes.extend(gene.copy() for gene in p2.genes[cut_point:])

        child2_genes = [gene.copy() for gene in p2.genes[:cut_point]]
        child2_genes.extend(gene.copy() for gene in p1.genes[cut_point:])

        child1 = Individual(genes=child1_genes, fitness=0.0)
        child2 = Individual(genes=child2_genes, fitness=0.0)
        return child1, child2
