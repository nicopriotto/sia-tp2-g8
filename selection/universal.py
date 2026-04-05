import numpy as np

from core.individual import Individual
from core.population import Population
from selection.roulette import _build_cumulative, _select_from_cumulative
from selection.selection_strategy import SelectionStrategy


class UniversalSelection(SelectionStrategy):
    def select(self, population: Population, k: int) -> list[Individual]:
        if k == 0:
            return []

        fitnesses = np.array([ind.fitness for ind in population.individuals])
        q = _build_cumulative(fitnesses)

        if q is None:
            indices = np.random.randint(0, len(population.individuals), size=k)
            return [population.individuals[i].copy() for i in indices]

        r = np.random.uniform(0, 1.0 / k)
        r_values = np.array([(r + j) / k for j in range(k)])
        indices = _select_from_cumulative(q, r_values)
        return [population.individuals[i].copy() for i in indices]
