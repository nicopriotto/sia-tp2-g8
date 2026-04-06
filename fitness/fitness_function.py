from abc import ABC, abstractmethod
import numpy as np


class FitnessFunction(ABC):
    """Clase base abstracta para funciones de fitness."""

    name: str  # "mse" o "mae", usado por GPURenderer

    @abstractmethod
    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Calcula el fitness entre una imagen generada y la imagen objetivo.

        Args:
            generated: Array numpy de shape (H, W, 4), dtype float32, valores en [0, 1].
                       Imagen generada por el renderer a partir del genotipo de un individuo.
            target: Array numpy de shape (H, W, 4), dtype float32, valores en [0, 1].
                    Imagen objetivo que el algoritmo intenta aproximar.

        Returns:
            float en el rango (0, 1], donde 1.0 = imagenes identicas.
            Mayor valor = mejor fitness.
        """
        ...
