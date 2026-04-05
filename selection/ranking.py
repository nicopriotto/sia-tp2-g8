import numpy as np

from core.individual import Individual
from core.population import Population
from selection.roulette import _build_cumulative, _select_from_cumulative
from selection.selection_strategy import SelectionStrategy


class RankingSelection(SelectionStrategy):
    def select(self, population: Population, k: int) -> list[Individual]:
        if k == 0:
            return []

        fitnesses = np.array([ind.fitness for ind in population.individuals])
        n = len(fitnesses)

        if n == 1:
            return [population.individuals[0].copy() for _ in range(k)]

        sorted_indices = np.argsort(-fitnesses)
        ranks = np.empty(n, dtype=int)
        for rank_pos, idx in enumerate(sorted_indices):
            ranks[idx] = rank_pos + 1

        pseudo_fitness = (n - ranks) / n

        q = _build_cumulative(pseudo_fitness)
        if q is None:
            indices = np.random.randint(0, n, size=k)
            return [population.individuals[i].copy() for i in indices]

        r_values = np.random.random(k)
        indices = _select_from_cumulative(q, r_values)
        return [population.individuals[i].copy() for i in indices]
