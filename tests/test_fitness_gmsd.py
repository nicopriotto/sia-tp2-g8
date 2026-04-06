import numpy as np
import pytest
from fitness.gmsd import GMSDFitness


@pytest.fixture
def gmsd():
    return GMSDFitness()


def test_gmsd_identical(gmsd):
    """Imagenes identicas -> fitness == 1.0."""
    img = np.random.rand(64, 64, 4).astype(np.float32)
    assert gmsd.compute(img, img) == pytest.approx(1.0)


def test_gmsd_range(gmsd):
    """Fitness siempre en (0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        a = rng.random((32, 32, 4), dtype=np.float32)
        b = rng.random((32, 32, 4), dtype=np.float32)
        f = gmsd.compute(a, b)
        assert 0.0 < f <= 1.0


def test_gmsd_symmetry(gmsd):
    """compute(A, B) == compute(B, A)."""
    rng = np.random.default_rng(123)
    a = rng.random((32, 32, 4), dtype=np.float32)
    b = rng.random((32, 32, 4), dtype=np.float32)
    assert gmsd.compute(a, b) == pytest.approx(gmsd.compute(b, a))


def test_gmsd_edge_sensitivity(gmsd):
    """GMSD penaliza mas los errores en bordes que MSE."""
    from fitness.mse import MSEFitness
    mse = MSEFitness()

    # Imagen uniforme gris
    uniform = np.full((64, 64, 4), 0.5, dtype=np.float32)
    uniform[:, :, 3] = 1.0

    # Imagen con bordes fuertes (patron de tablero)
    edges = uniform.copy()
    for i in range(64):
        for j in range(64):
            if (i // 8 + j // 8) % 2 == 0:
                edges[i, j, :3] = 0.0
            else:
                edges[i, j, :3] = 1.0

    # GMSD deberia dar fitness mas bajo (mas distorsion percibida)
    # que MSE para este tipo de error estructural
    gmsd_fitness = gmsd.compute(uniform, edges)
    mse_fitness = mse.compute(uniform, edges)
    assert gmsd_fitness < mse_fitness
