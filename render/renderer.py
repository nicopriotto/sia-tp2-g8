from abc import ABC, abstractmethod
import numpy as np


class Renderer(ABC):
    """Clase base abstracta para renderers que convierten genes a imagenes."""

    @abstractmethod
    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """
        Renderiza una lista de genes como una imagen RGBA.

        Args:
            genes: Lista de TriangleGene a renderizar.
            width: Ancho de la imagen en pixeles.
            height: Alto de la imagen en pixeles.

        Returns:
            Array numpy de shape (height, width, 4), dtype float32, valores en [0, 1].
        """
        ...
