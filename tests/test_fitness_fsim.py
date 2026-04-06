import time
import numpy as np
import pytest
from fitness.fsim import FSIMFitness


@pytest.fixture
def fsim():
    return FSIMFitness()


def test_fsim_identical(fsim):
    """Imagenes identicas -> fitness > 0.99."""
    img = np.random.rand(64, 64, 4).astype(np.float32)
    assert fsim.compute(img, img) > 0.99


def test_fsim_range(fsim):
    """Fitness siempre en (0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        a = rng.random((64, 64, 4), dtype=np.float32)
        b = rng.random((64, 64, 4), dtype=np.float32)
        f = fsim.compute(a, b)
        assert 0.0 <= f <= 1.0


def test_fsim_symmetry(fsim):
    """compute(A, B) ~= compute(B, A)."""
    rng = np.random.default_rng(123)
    a = rng.random((64, 64, 4), dtype=np.float32)
    b = rng.random((64, 64, 4), dtype=np.float32)
    assert fsim.compute(a, b) == pytest.approx(fsim.compute(b, a), abs=1e-6)


def test_fsim_edge_sensitivity(fsim):
    """FSIM es mas sensible a bordes que MSE."""
    from fitness.mse import MSEFitness
    mse = MSEFitness()

    # Imagen uniforme gris
    uniform = np.full((64, 64, 4), 0.5, dtype=np.float32)
    uniform[:, :, 3] = 1.0

    # Imagen con bordes fuertes (tablero)
    edges = uniform.copy()
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                edges[i, j, :3] = 0.0
            else:
                edges[i, j, :3] = 1.0

    fsim_f = fsim.compute(uniform, edges)
    mse_f = mse.compute(uniform, edges)
    # FSIM deberia penalizar mas los bordes
    assert fsim_f < mse_f


def test_fsim_performance(fsim):
    """128x128 debe calcular en menos de 5 segundos."""
    rng = np.random.default_rng(55)
    a = rng.random((128, 128, 4), dtype=np.float32)
    b = rng.random((128, 128, 4), dtype=np.float32)
    start = time.time()
    fsim.compute(a, b)
    elapsed = time.time() - start
    assert elapsed < 5.0
