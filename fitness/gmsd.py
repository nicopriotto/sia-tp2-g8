import numpy as np
from fitness.fitness_function import FitnessFunction


class GMSDFitness(FitnessFunction):
    """
    GMSD: Gradient Magnitude Similarity Deviation.

    Calcula la desviacion estandar del mapa de similitud de gradientes.
    Los bordes y estructuras localizadas tienen mas peso que en MSE/MAE,
    correlacionando mejor con la percepcion humana.

    Referencia: Xue et al., IEEE TIP 2014. https://arxiv.org/abs/1308.3052
    """

    name = "gmsd"

    # Filtros Prewitt 3x3 normalizados
    _Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 3.0
    _Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 3.0

    # Constante de estabilidad del paper (T = 170/255^2)
    _C = (170.0 / 255.0) ** 2

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        gen_gray = self._to_gray(generated[:, :, :3])
        tgt_gray = self._to_gray(target[:, :, :3])

        gm_gen = self._gradient_magnitude(gen_gray)
        gm_tgt = self._gradient_magnitude(tgt_gray)

        gms = (2.0 * gm_gen * gm_tgt + self._C) / (
            gm_gen ** 2 + gm_tgt ** 2 + self._C
        )

        gmsd = float(np.std(gms))
        return 1.0 / (1.0 + gmsd)

    def _to_gray(self, rgb: np.ndarray) -> np.ndarray:
        """Luminancia ITU-R BT.601."""
        return (0.299 * rgb[:, :, 0] +
                0.587 * rgb[:, :, 1] +
                0.114 * rgb[:, :, 2]).astype(np.float32)

    def _conv2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Convolucion 2D con kernel 3x3, padding reflect, numpy puro."""
        padded = np.pad(img, 1, mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        return (windows * kernel).sum(axis=(-1, -2)).astype(np.float32)

    def _gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        gx = self._conv2d(gray, self._Kx)
        gy = self._conv2d(gray, self._Ky)
        return np.sqrt(gx ** 2 + gy ** 2)
