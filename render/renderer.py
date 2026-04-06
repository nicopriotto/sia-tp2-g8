from abc import ABC, abstractmethod
import numpy as np


class Renderer(ABC):
    """Clase base abstracta para renderers que convierten genes a imagenes."""

    def compute_fitness(self, genes: np.ndarray, fitness_type: str = "mse", gene_type: str = "triangle") -> float | None:
        """Calcula fitness en GPU si el renderer lo soporta. Retorna None si no."""
        return None

    @abstractmethod
    def render(self, genes: np.ndarray, width: int, height: int, gene_type: str = "triangle") -> np.ndarray:
        """
        Renderiza un array de genes como una imagen RGBA.

        Args:
            genes: Array numpy (n_genes, 11) con los genes a renderizar.
            width: Ancho de la imagen en pixeles.
            height: Alto de la imagen en pixeles.
            gene_type: Tipo de gen ("triangle" o "ellipse").

        Returns:
            Array numpy de shape (height, width, 4), dtype float32, valores en [0, 1].
        """
        ...
