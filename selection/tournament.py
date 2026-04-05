import numpy as np

from core.individual import Individual
from core.population import Population
from selection.selection_strategy import SelectionStrategy


class DeterministicTournamentSelection(SelectionStrategy):
    def __init__(self, m: int):
        if m < 1:
            raise ValueError(f"El tamanio del torneo debe ser >= 1, recibido: {m}")
        self.m = m

    def select(self, population: Population, k: int, generation: int = 0) -> list[Individual]:
        if k == 0:
            return []

        individuals = population.individuals
        n = len(individuals)
        selected = []

        for _ in range(k):
            tournament_indices = np.random.randint(0, n, size=self.m)
            tournament_fitnesses = np.array([individuals[i].fitness for i in tournament_indices])
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(individuals[winner_idx].copy())

        return selected


class ProbabilisticTournamentSelection(SelectionStrategy):
    def __init__(self, threshold: float):
        if not (0.5 <= threshold <= 1.0):
            raise ValueError(f"El threshold debe estar en [0.5, 1.0], recibido: {threshold}")
        self.threshold = threshold

    def select(self, population: Population, k: int, generation: int = 0) -> list[Individual]:
        if k == 0:
            return []

        individuals = population.individuals
        n = len(individuals)
        selected = []

        for _ in range(k):
            idx1, idx2 = np.random.randint(0, n, size=2)
            ind1 = individuals[idx1]
            ind2 = individuals[idx2]

            if ind1.fitness >= ind2.fitness:
                more_fit, less_fit = ind1, ind2
            else:
                more_fit, less_fit = ind2, ind1

            if np.random.random() < self.threshold:
                selected.append(more_fit.copy())
            else:
                selected.append(less_fit.copy())

        return selected
