import numpy as np
from fitness.fitness_function import FitnessFunction


class LinearMSEFitness(FitnessFunction):
    """Funcion de fitness lineal basada en MSE: fitness = 1 - MSE."""

    name = "linear_mse"

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Calcula fitness como 1 - MSE sobre los canales RGB.

        Con imagenes normalizadas a [0, 1], MSE esta en [0, 1], por lo que
        fitness esta en [0, 1] donde 1.0 = imagenes identicas.
        """
        gen_rgb = generated[:, :, :3]
        tgt_rgb = target[:, :, :3]
        mse = np.mean((gen_rgb - tgt_rgb) ** 2)
        fitness = 1.0 - mse
        return float(max(fitness, 0.0))
