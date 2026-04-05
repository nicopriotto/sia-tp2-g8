from abc import ABC, abstractmethod

from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy


class SurvivalStrategy(ABC):
    """Clase base abstracta para estrategias de supervivencia."""

    @abstractmethod
    def apply(
        self,
        current: Population,
        children: list[Individual],
        selection: SelectionStrategy,
    ) -> Population:
        """Aplica la estrategia de supervivencia y retorna la nueva poblacion."""
        raise NotImplementedError
