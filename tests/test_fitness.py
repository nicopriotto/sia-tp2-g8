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


import pytest
from fitness.mae import MAEFitness


def test_mae_identical_images():
    rng = np.random.default_rng(42)
    img = rng.random((10, 10, 4), dtype=np.float32)
    assert MAEFitness().compute(img, img) == pytest.approx(1.0)


def test_mae_range():
    rng = np.random.default_rng(99)
    fn = MAEFitness()
    for _ in range(100):
        a = rng.random((10, 10, 4), dtype=np.float32)
        b = rng.random((10, 10, 4), dtype=np.float32)
        f = fn.compute(a, b)
        assert 0.0 < f <= 1.0


def test_mse_penalizes_outliers_more():
    a = np.zeros((10, 10, 4), dtype=np.float32)
    a[:, :, 3] = 1.0
    b = a.copy()
    b[0, 0, :3] = 1.0  # single white pixel outlier

    mse_fitness = MSEFitness().compute(a, b)
    mae_fitness = MAEFitness().compute(a, b)
    assert mse_fitness <= mae_fitness


def test_mae_symmetry():
    rng = np.random.default_rng(7)
    a = rng.random((10, 10, 4), dtype=np.float32)
    b = rng.random((10, 10, 4), dtype=np.float32)
    fn = MAEFitness()
    assert fn.compute(a, b) == pytest.approx(fn.compute(b, a))
