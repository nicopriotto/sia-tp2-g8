from crossover.crossover_operator import CrossoverOperator
from core.individual import Individual


class ArithmeticCrossover(CrossoverOperator):
    """Crossover aritmetico: interpola genes entre dos padres."""

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: Factor de interpolacion en [0, 1].
                   0.5 = promedio de ambos padres.
                   1.0 = hijo1 es copia de p1, hijo2 es copia de p2.
        """
        self.alpha = alpha

    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Aplica crossover aritmetico: interpola cada gen entre ambos padres."""
        n = len(p1.genes)
        assert len(p2.genes) == n, "Padres deben tener la misma cantidad de genes"

        genes1 = [g1.blend(g2, self.alpha) for g1, g2 in zip(p1.genes, p2.genes)]
        genes2 = [g2.blend(g1, self.alpha) for g1, g2 in zip(p1.genes, p2.genes)]

        child1 = Individual(genes=genes1)  # fitness = 0.0
        child2 = Individual(genes=genes2)  # fitness = 0.0
        return child1, child2
