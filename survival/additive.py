from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy
from survival.survival_strategy import SurvivalStrategy


class AdditiveSurvival(SurvivalStrategy):
    """Supervivencia que combina padres e hijos en un unico pool."""

    def apply(
        self,
        current: Population,
        children: list[Individual],
        selection: SelectionStrategy,
    ) -> Population:
        """Selecciona N sobrevivientes del pool padres+hijos."""
        population_size = len(current.individuals)
        pool_population = Population(individuals=current.individuals + children)
        selected = selection.select(pool_population, population_size)
        return Population(individuals=selected)
