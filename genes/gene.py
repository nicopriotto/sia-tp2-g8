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

    @abstractmethod
    def blend(self, other: "Gene", alpha: float) -> "Gene":
        """Interpola atributos: resultado[i] = alpha * self[i] + (1-alpha) * other[i].

        Args:
            other: Otro gen del mismo tipo.
            alpha: Factor de interpolacion en [0, 1].
                   alpha=1.0 retorna copia de self, alpha=0.0 retorna copia de other.

        Returns:
            Gen nuevo con atributos interpolados y clampeados al rango valido.
        """
        ...

    @abstractmethod
    def mutate_gaussian(self, sigma: float) -> "Gene":
        """Perturba cada atributo sumando delta ~ N(0, sigma), clampeando al rango valido.

        Args:
            sigma: Desviacion estandar de la distribucion normal.

        Returns:
            Gen nuevo con atributos perturbados.
        """
        ...
