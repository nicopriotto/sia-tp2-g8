from abc import ABC, abstractmethod

from core.individual import Individual


class MutationOperator(ABC):
    """Clase base abstracta para operadores de mutacion."""

    @abstractmethod
    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        """Aplica mutacion a un individuo y retorna el resultado."""
        raise NotImplementedError
