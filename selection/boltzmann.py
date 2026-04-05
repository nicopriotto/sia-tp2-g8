import numpy as np

from core.individual import Individual
from core.population import Population
from selection.roulette import _build_cumulative, _select_from_cumulative
from selection.selection_strategy import SelectionStrategy


class BoltzmannSelection(SelectionStrategy):
    def __init__(self, t0: float, tc: float, k: float):
        self.t0 = t0
        self.tc = tc
        self.k = k

    def select(self, population: Population, k: int, generation: int = 0) -> list[Individual]:
        if k == 0:
            return []

        T = self.tc + (self.t0 - self.tc) * np.exp(-self.k * generation)
        T = max(T, 1e-10)

        fitnesses = np.array([ind.fitness for ind in population.individuals])
        max_fitness = np.max(fitnesses)
        exp_vals = np.exp((fitnesses - max_fitness) / T)

        mean_exp = np.mean(exp_vals)
        if mean_exp == 0:
            indices = np.random.randint(0, len(population.individuals), size=k)
            return [population.individuals[i].copy() for i in indices]

        pseudo_fitness = exp_vals / mean_exp

        q = _build_cumulative(pseudo_fitness)
        if q is None:
            indices = np.random.randint(0, len(population.individuals), size=k)
            return [population.individuals[i].copy() for i in indices]

        r_values = np.random.random(k)
        indices = _select_from_cumulative(q, r_values)
        return [population.individuals[i].copy() for i in indices]
