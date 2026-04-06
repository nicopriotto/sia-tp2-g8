import numpy as np

from fitness.fitness_function import FitnessFunction


class MAEFitness(FitnessFunction):
    name = "mae"

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        mae = np.mean(np.abs(generated[:, :, :3] - target[:, :, :3]))
        return float(1.0 / (1.0 + mae))
