from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy
from survival.survival_strategy import SurvivalStrategy


class ExclusiveSurvival(SurvivalStrategy):
    def apply(
        self,
        current: Population,
        children: list[Individual],
        selection: SelectionStrategy,
    ) -> Population:
        n = len(current.individuals)
        k = len(children)

        if k >= n:
            children_pop = Population(individuals=children)
            selected = selection.select(children_pop, n)
            return Population(individuals=selected)
        else:
            survivors = list(children)
            if n - k > 0:
                survivors += selection.select(current, n - k)
            return Population(individuals=survivors)
