import bisect

import numpy as np

from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy


def _build_cumulative(fitnesses: np.ndarray) -> np.ndarray | None:
    """Construye el array de probabilidades acumuladas a partir de fitness.

    Retorna None si todos los fitness son iguales (seleccion uniforme).
    """
    total = fitnesses.sum()
    if total == 0 or np.all(fitnesses == fitnesses[0]):
        return None
    p = fitnesses / total
    q = np.cumsum(p)
    q[-1] = 1.0
    return q


def _select_from_cumulative(q: np.ndarray, r_values: np.ndarray) -> list[int]:
    """Dado un array de acumulados y valores r, retorna los indices seleccionados."""
    return [bisect.bisect_left(q, r) for r in r_values]


class RouletteSelection(SelectionStrategy):
    def select(self, population: Population, k: int, generation: int = 0) -> list[Individual]:
        if k == 0:
            return []

        fitnesses = np.array([ind.fitness for ind in population.individuals])
        q = _build_cumulative(fitnesses)

        if q is None:
            indices = np.random.randint(0, len(population.individuals), size=k)
            return [population.individuals[i].copy() for i in indices]

        r_values = np.random.random(k)
        indices = _select_from_cumulative(q, r_values)
        return [population.individuals[i].copy() for i in indices]
