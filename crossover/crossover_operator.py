from abc import ABC, abstractmethod

from core.individual import Individual


class CrossoverOperator(ABC):
    """Clase base abstracta para operadores de crossover."""

    @abstractmethod
    def crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        """Aplica crossover entre dos padres y genera dos hijos."""
        raise NotImplementedError
