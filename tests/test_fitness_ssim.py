import numpy as np
import pytest
from fitness.ssim import SSIMFitness


@pytest.fixture
def ssim():
    return SSIMFitness()


def test_ssim_identical(ssim):
    """Imagenes identicas -> fitness == 1.0."""
    img = np.random.rand(64, 64, 4).astype(np.float32)
    assert ssim.compute(img, img) == pytest.approx(1.0)


def test_ssim_range(ssim):
    """Fitness siempre en (0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(50):
        a = rng.random((32, 32, 4), dtype=np.float32)
        b = rng.random((32, 32, 4), dtype=np.float32)
        f = ssim.compute(a, b)
        assert 0.0 <= f <= 1.0


def test_ssim_symmetry(ssim):
    """compute(A, B) == compute(B, A)."""
    rng = np.random.default_rng(123)
    a = rng.random((32, 32, 4), dtype=np.float32)
    b = rng.random((32, 32, 4), dtype=np.float32)
    assert ssim.compute(a, b) == pytest.approx(ssim.compute(b, a))


def test_ssim_small_image(ssim):
    """Imagen 4x4 (menor que window_size) no debe crashear."""
    rng = np.random.default_rng(77)
    a = rng.random((4, 4, 4), dtype=np.float32)
    b = rng.random((4, 4, 4), dtype=np.float32)
    f = ssim.compute(a, b)
    assert 0.0 <= f <= 1.0


def test_ssim_better_than_mse_perceptually(ssim):
    """SSIM distingue error concentrado vs distribuido mejor que MSE."""
    from fitness.mse import MSEFitness
    mse = MSEFitness()

    base = np.full((64, 64, 4), 0.5, dtype=np.float32)
    base[:, :, 3] = 1.0

    # Error concentrado en un parche 16x16
    concentrated = base.copy()
    concentrated[24:40, 24:40, :3] = 0.0

    # Error distribuido uniformemente con misma magnitud total
    diff_total = np.sum((base[:, :, :3] - concentrated[:, :, :3]) ** 2)
    per_pixel = np.sqrt(diff_total / (64 * 64 * 3))
    distributed = base.copy()
    distributed[:, :, :3] -= per_pixel

    # MSE deberia ser similar para ambos
    mse_conc = mse.compute(base, concentrated)
    mse_dist = mse.compute(base, distributed)
    assert mse_conc == pytest.approx(mse_dist, abs=0.01)

    # SSIM deberia distinguirlos: error concentrado es peor perceptualmente
    ssim_conc = ssim.compute(base, concentrated)
    ssim_dist = ssim.compute(base, distributed)
    assert ssim_conc < ssim_dist
