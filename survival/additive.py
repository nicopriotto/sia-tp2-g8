from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy
from survival.survival_strategy import SurvivalStrategy


class AdditiveSurvival(SurvivalStrategy):
    """Supervivencia que combina padres e hijos y selecciona los N mejores."""

    def apply(
        self,
        current: Population,
        children: list[Individual],
        selection: SelectionStrategy,
    ) -> Population:
        """Selecciona los N mejores del pool padres+hijos por truncamiento."""
        population_size = len(current.individuals)
        pool = current.individuals + children
        pool.sort(key=lambda ind: ind.fitness, reverse=True)
        survivors = pool[:population_size]
        return Population(individuals=survivors)
