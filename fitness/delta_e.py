import numpy as np

from fitness.fitness_function import FitnessFunction


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convierte imagen RGB float32 [0,1] a CIE Lab.

    Args:
        rgb: Array (H, W, 3) float32 en [0, 1].

    Returns:
        Array (H, W, 3) float32 en espacio CIE Lab.
        L en [0, 100], a y b en aprox [-128, 127].
    """
    # Paso 1: RGB lineal (deshacer sRGB gamma)
    mask = rgb > 0.04045
    linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    # Paso 2: RGB lineal -> XYZ (iluminante D65)
    # Matriz de conversion sRGB -> XYZ
    m = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = linear @ m.T

    # Paso 3: XYZ -> Lab
    # Referencia D65: Xn=0.95047, Yn=1.0, Zn=1.08883
    ref = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz_n = xyz / ref

    delta = 6.0 / 29.0
    delta_sq = delta ** 2
    delta_cb = delta ** 3

    mask = xyz_n > delta_cb
    f = np.where(mask, np.cbrt(xyz_n), xyz_n / (3.0 * delta_sq) + 4.0 / 29.0)

    L = 116.0 * f[:, :, 1] - 16.0
    a = 500.0 * (f[:, :, 0] - f[:, :, 1])
    b = 200.0 * (f[:, :, 1] - f[:, :, 2])

    return np.stack([L, a, b], axis=-1)


class DeltaEFitness(FitnessFunction):
    """Funcion de fitness basada en Delta E76 (CIE76) en espacio CIE Lab."""

    name = "delta_e"

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        """
        Calcula fitness como 1 / (1 + mean_deltaE).

        Delta E76 es la distancia euclidiana en CIE Lab, perceptualmente
        uniforme: cada unidad de mejora corresponde a una mejora visible.
        """
        gen_lab = _rgb_to_lab(generated[:, :, :3])
        tgt_lab = _rgb_to_lab(target[:, :, :3])

        # Delta E76: distancia euclidiana por pixel en Lab
        delta_e = np.sqrt(np.sum((gen_lab - tgt_lab) ** 2, axis=-1))
        mean_delta_e = np.mean(delta_e)

        fitness = 1.0 / (1.0 + mean_delta_e)
        return float(fitness)
