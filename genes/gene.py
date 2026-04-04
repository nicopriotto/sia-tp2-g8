from abc import ABC, abstractmethod


class Gene(ABC):
    """Clase base abstracta para un gen del algoritmo genetico."""

    @abstractmethod
    def mutate_replace(self) -> "Gene":
        """Retorna un gen completamente nuevo con valores aleatorios."""
        ...

    @abstractmethod
    def mutate_delta(self, strength: float) -> "Gene":
        """Retorna un gen con atributos perturbados por un delta proporcional a strength."""
        ...

    @abstractmethod
    def copy(self) -> "Gene":
        """Retorna una copia independiente del gen (sin referencias compartidas)."""
        ...
