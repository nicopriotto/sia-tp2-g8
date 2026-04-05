from math import ceil

from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy


class EliteSelection(SelectionStrategy):
    """Seleccion deterministica que prioriza a los individuos mas aptos."""

    def select(self, population: Population, k: int) -> list[Individual]:
        """Selecciona individuos segun la formula elite 0-indexed."""
        sorted_individuals = sorted(
            population.individuals,
            key=lambda individual: individual.fitness,
            reverse=True,
        )
        population_size = len(sorted_individuals)
        selected: list[Individual] = []

        for index, individual in enumerate(sorted_individuals):
            repetitions = ceil((k - index) / population_size)
            if repetitions <= 0:
                break
            selected.extend(individual.copy() for _ in range(repetitions))

        return selected[:k]
