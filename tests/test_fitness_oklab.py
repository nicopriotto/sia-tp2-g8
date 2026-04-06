import numpy as np
import pytest
from fitness.oklab import OklabMSEFitness


@pytest.fixture
def oklab():
    return OklabMSEFitness()


def test_oklab_identical(oklab):
    """Imagenes identicas -> fitness == 1.0."""
    img = np.random.rand(64, 64, 4).astype(np.float32)
    assert oklab.compute(img, img) == pytest.approx(1.0)


def test_oklab_range(oklab):
    """Fitness siempre en (0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        a = rng.random((32, 32, 4), dtype=np.float32)
        b = rng.random((32, 32, 4), dtype=np.float32)
        f = oklab.compute(a, b)
        assert 0.0 < f <= 1.0


def test_oklab_symmetry(oklab):
    """compute(A, B) == compute(B, A)."""
    rng = np.random.default_rng(123)
    a = rng.random((32, 32, 4), dtype=np.float32)
    b = rng.random((32, 32, 4), dtype=np.float32)
    assert oklab.compute(a, b) == pytest.approx(oklab.compute(b, a))


def test_oklab_perceptual_blue(oklab):
    """Oklab diferencia errores en azules que MSE en RGB no distingue."""
    from fitness.mse import MSEFitness
    mse = MSEFitness()

    # Imagen base gris medio
    base = np.full((32, 32, 4), 0.5, dtype=np.float32)
    base[:, :, 3] = 1.0

    # Perturbacion A: error en canal azul
    blue_err = base.copy()
    blue_err[:, :, 2] = 0.7  # +0.2 en azul

    # Perturbacion B: error en canal verde (misma magnitud)
    green_err = base.copy()
    green_err[:, :, 1] = 0.7  # +0.2 en verde

    # MSE en RGB las ve iguales
    mse_blue = mse.compute(base, blue_err)
    mse_green = mse.compute(base, green_err)
    assert mse_blue == pytest.approx(mse_green, abs=1e-6)

    # Oklab las diferencia (azul y verde tienen distinto peso perceptual)
    oklab_blue = oklab.compute(base, blue_err)
    oklab_green = oklab.compute(base, green_err)
    assert oklab_blue != pytest.approx(oklab_green, abs=1e-4)
