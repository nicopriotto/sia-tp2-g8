import numpy as np
import pytest
from fitness.mse import MSEFitness


def test_mse_identical_images():
    rng = np.random.default_rng(42)
    img = rng.random((50, 50, 4), dtype=np.float32)
    fitness_fn = MSEFitness()
    assert fitness_fn.compute(img, img) == pytest.approx(1.0)


def test_mse_opposite_images():
    white = np.ones((50, 50, 4), dtype=np.float32)
    black = np.zeros((50, 50, 4), dtype=np.float32)
    black[:, :, 3] = 1.0  # alpha opaco
    fitness_fn = MSEFitness()
    fitness = fitness_fn.compute(white, black)
    # MSE de blanco vs negro sobre RGB = 1.0, fitness = 1/(1+1) = 0.5
    assert fitness <= 0.6


def test_mse_range():
    rng = np.random.default_rng(123)
    fitness_fn = MSEFitness()
    for _ in range(20):
        a = rng.random((30, 30, 4), dtype=np.float32)
        b = rng.random((30, 30, 4), dtype=np.float32)
        fitness = fitness_fn.compute(a, b)
        assert 0.0 < fitness <= 1.0


def test_mse_symmetry():
    rng = np.random.default_rng(456)
    a = rng.random((30, 30, 4), dtype=np.float32)
    b = rng.random((30, 30, 4), dtype=np.float32)
    fitness_fn = MSEFitness()
    assert fitness_fn.compute(a, b) == pytest.approx(fitness_fn.compute(b, a))
