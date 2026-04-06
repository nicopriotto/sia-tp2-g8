import random

import numpy as np

from core.individual import Individual
from genes import gene_layout
from mutation.mutation_operator import MutationOperator


class NonUniformMutation(MutationOperator):
    def __init__(self, mutation_rate: float, b: float = 5.0):
        self.mutation_rate = mutation_rate
        self.b = b

    def mutate(self, individual: Individual, generation: int, max_generations: int) -> Individual:
        mutated = individual.copy()

        if random.random() < self.mutation_rate:
            if max_generations == 0:
                strength = 1.0
            else:
                strength = (1 - generation / max_generations) ** self.b

            idx = random.randint(0, mutated.genes.shape[0] - 1)
            row = mutated.genes[idx].copy()
            gt = mutated.gene_type
            layout = gene_layout.LAYOUTS[gt]

            current_strength = strength
            for _ in range(10):
                delta = np.random.uniform(-current_strength, current_strength, size=gene_layout.N_COLS)
                delta[6:9] *= 255  # scale RGB deltas
                delta[10] = 0  # don't perturb active
                if gt == "ellipse":
                    delta[5] = 0  # don't perturb padding
                candidate = row + delta
                np.clip(candidate, layout["low"], layout["high"], out=candidate)
                candidate[6:9] = np.round(candidate[6:9])

                if gt != "triangle" or not gene_layout.is_degenerate(candidate):
                    # 5% chance to flip active
                    if random.random() < 0.05:
                        candidate[10] = 1.0 - candidate[10]
                    mutated.genes[idx] = candidate
                    mutated.fitness_valid = False
                    return mutated

                current_strength *= 2

            # All attempts produced degenerate triangles, don't mutate
        return mutated
