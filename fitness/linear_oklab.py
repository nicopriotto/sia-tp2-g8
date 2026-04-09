import numpy as np
from fitness.fitness_function import FitnessFunction


class LinearOklabFitness(FitnessFunction):
    """
    MSE en espacio Oklab con mapeo lineal: fitness = 1 - MSE.

    Combina la uniformidad perceptual de Oklab (distancias proporcionales
    a la diferencia visual percibida) con el mapeo lineal que mantiene
    presión selectiva uniforme desde el inicio de la evolución.
    """

    name = "linear_oklab"

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
        return max(1.0 - mse, 0.0)

    def _srgb_to_oklab(self, rgb: np.ndarray) -> np.ndarray:
        rgb64 = rgb.astype(np.float64)
        linear = np.where(
            rgb64 <= 0.04045,
            rgb64 / 12.92,
            ((rgb64 + 0.055) / 1.055) ** 2.4,
        )
        lms = linear @ self._M1.T
        lms_cbrt = np.cbrt(np.clip(lms, 0.0, None))
        return lms_cbrt @ self._M2.T
