import numpy as np
from fitness.fitness_function import FitnessFunction


class OklabMSEFitness(FitnessFunction):
    """
    MSE calculado en espacio de color Oklab (perceptualmente uniforme).

    Distancias iguales en Oklab corresponden a diferencias percibidas iguales,
    a diferencia de sRGB donde el mismo error numerico puede ser invisible
    en zonas oscuras y muy visible en zonas claras.

    Referencia: Ottosson (2020). https://bottosson.github.io/posts/oklab/
    """

    name = "oklab"

    _M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ], dtype=np.float64)

    _M2 = np.array([
        [0.2104542553,  0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050,  0.4505937099],
        [0.0259040371,  0.7827717662, -0.8086757660],
    ], dtype=np.float64)

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        lab_gen = self._srgb_to_oklab(generated[:, :, :3])
        lab_tgt = self._srgb_to_oklab(target[:, :, :3])
        mse = float(np.mean((lab_gen - lab_tgt) ** 2))
        return 1.0 / (1.0 + mse)

    def _srgb_to_oklab(self, rgb: np.ndarray) -> np.ndarray:
        rgb64 = rgb.astype(np.float64)
        # Gamma decode
        linear = np.where(
            rgb64 <= 0.04045,
            rgb64 / 12.92,
            ((rgb64 + 0.055) / 1.055) ** 2.4,
        )
        # Linear RGB -> LMS
        lms = linear @ self._M1.T
        # Cube root (clip para evitar dominio negativo por errores numericos)
        lms_cbrt = np.cbrt(np.clip(lms, 0.0, None))
        # LMS^(1/3) -> Lab
        return lms_cbrt @ self._M2.T
