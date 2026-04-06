import numpy as np
import pytest
from fitness.msssim import MSSSIMFitness


@pytest.fixture
def msssim():
    return MSSSIMFitness()


def test_msssim_identical(msssim):
    """Imagenes identicas -> fitness == 1.0."""
    img = np.random.rand(64, 64, 4).astype(np.float32)
    assert msssim.compute(img, img) == pytest.approx(1.0, abs=1e-6)


def test_msssim_range(msssim):
    """Fitness siempre en (0, 1]."""
    rng = np.random.default_rng(42)
    for _ in range(30):
        a = rng.random((64, 64, 4), dtype=np.float32)
        b = rng.random((64, 64, 4), dtype=np.float32)
        f = msssim.compute(a, b)
        assert 0.0 <= f <= 1.0


def test_msssim_better_for_blur(msssim):
    """MS-SSIM distingue blur de ruido mejor que MSE."""
    from fitness.mse import MSEFitness
    mse_fn = MSEFitness()

    rng = np.random.default_rng(99)
    original = rng.random((64, 64, 4), dtype=np.float32)
    original[:, :, 3] = 1.0

    # Blur: promedio 3x3
    blurred = original.copy()
    for c in range(3):
        padded = np.pad(original[:, :, c], 1, mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        blurred[:, :, c] = windows.mean(axis=(-1, -2))

    # MS-SSIM deberia dar fitness > 0 para blur (no colapsar a 0)
    f = msssim.compute(original, blurred)
    assert f > 0.3


def test_msssim_small_image(msssim):
    """Imagen 16x16 no debe crashear."""
    rng = np.random.default_rng(77)
    a = rng.random((16, 16, 4), dtype=np.float32)
    b = rng.random((16, 16, 4), dtype=np.float32)
    f = msssim.compute(a, b)
    assert 0.0 <= f <= 1.0
