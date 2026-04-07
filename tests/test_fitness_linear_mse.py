import numpy as np
import pytest

from fitness.linear_mse import LinearMSEFitness
from fitness.mse import MSEFitness


class TestLinearMSEFitness:
    def setup_method(self):
        self.fitness = LinearMSEFitness()

    def test_linear_identical_images(self):
        """Imagenes identicas dan fitness = 1.0."""
        img = np.random.rand(50, 50, 4).astype(np.float32)
        assert self.fitness.compute(img, img) == pytest.approx(1.0)

    def test_linear_opposite_images(self):
        """Blanco vs negro: MSE_rgb = 1.0, fitness = 0.0."""
        white = np.ones((50, 50, 4), dtype=np.float32)
        black = np.zeros((50, 50, 4), dtype=np.float32)
        black[:, :, 3] = 1.0  # alpha opaco
        assert self.fitness.compute(white, black) == pytest.approx(0.0)

    def test_linear_range(self):
        """Fitness siempre en [0, 1] para imagenes aleatorias."""
        for _ in range(20):
            a = np.random.rand(30, 30, 4).astype(np.float32)
            b = np.random.rand(30, 30, 4).astype(np.float32)
            f = self.fitness.compute(a, b)
            assert 0.0 <= f <= 1.0

    def test_linear_symmetry(self):
        """compute(a, b) == compute(b, a)."""
        a = np.random.rand(30, 30, 4).astype(np.float32)
        b = np.random.rand(30, 30, 4).astype(np.float32)
        assert self.fitness.compute(a, b) == pytest.approx(self.fitness.compute(b, a))

    def test_linear_vs_inverse_discrimination(self):
        """LinearMSE discrimina mejor que 1/(1+MSE) entre individuos."""
        target = np.zeros((10, 10, 4), dtype=np.float32)
        target[:, :, 3] = 1.0

        # Crear 3 imagenes con MSE creciente
        imgs = []
        for mse_val in [0.1, 0.5, 0.9]:
            img = np.full((10, 10, 4), np.sqrt(mse_val), dtype=np.float32)
            img[:, :, 3] = 1.0
            imgs.append(img)

        linear = LinearMSEFitness()
        inverse = MSEFitness()

        linear_fits = [linear.compute(img, target) for img in imgs]
        inverse_fits = [inverse.compute(img, target) for img in imgs]

        # LinearMSE tiene mayor rango de diferencias
        linear_range = linear_fits[0] - linear_fits[2]
        inverse_range = inverse_fits[0] - inverse_fits[2]
        assert linear_range > inverse_range
