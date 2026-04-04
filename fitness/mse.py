import numpy as np
from fitness.fitness_function import FitnessFunction


class MSEFitness(FitnessFunction):
    """Funcion de fitness basada en Mean Squared Error (MSE)."""

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Calcula fitness como 1 / (1 + MSE) sobre los canales RGB.

        El MSE se calcula solo sobre los 3 primeros canales (RGB), ignorando
        el canal alpha (indice 3).
        """
        gen_rgb = generated[:, :, :3]
        tgt_rgb = target[:, :, :3]

        mse = np.mean((gen_rgb - tgt_rgb) ** 2)

        fitness = 1.0 / (1.0 + mse)
        return float(fitness)
