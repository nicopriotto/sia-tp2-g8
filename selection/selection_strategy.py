from abc import ABC, abstractmethod

from core.individual import Individual
from core.population import Population


class SelectionStrategy(ABC):
    """Clase base abstracta para estrategias de seleccion."""

    @abstractmethod
    def select(self, population: Population, k: int) -> list[Individual]:
        """Selecciona k individuos de la poblacion.

        Args:
            population: Poblacion de donde seleccionar.
            k: Cantidad de individuos a seleccionar.

        Returns:
            Lista de k copias de individuos seleccionados.
            NUNCA devolver referencias directas a los individuos de la poblacion.
        """
        raise NotImplementedError
