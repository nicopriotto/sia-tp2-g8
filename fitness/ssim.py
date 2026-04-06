import numpy as np
from fitness.fitness_function import FitnessFunction


class SSIMFitness(FitnessFunction):
    """
    SSIM: Structural Similarity Index.

    Mide similitud en terminos de luminancia, contraste y estructura,
    correlacionando mejor con la percepcion humana que MSE o MAE.

    Se calcula en bloques no solapados de window_size x window_size,
    promediando el SSIM por bloque sobre los 3 canales RGB.
    """

    name = "ssim"

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self._C1 = 0.01 ** 2
        self._C2 = 0.03 ** 2

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        gen = generated[:, :, :3].astype(np.float32)
        tgt = target[:, :, :3].astype(np.float32)

        H, W = gen.shape[:2]
        ws = self.window_size

        H_c = (H // ws) * ws
        W_c = (W // ws) * ws
        if H_c == 0 or W_c == 0:
            return self._ssim_global(gen, tgt)

        gen = gen[:H_c, :W_c]
        tgt = tgt[:H_c, :W_c]

        n_h = H_c // ws
        n_w = W_c // ws

        # Reshape a bloques: (n_h, n_w, ws, ws, 3)
        gen_b = gen.reshape(n_h, ws, n_w, ws, 3).transpose(0, 2, 1, 3, 4)
        tgt_b = tgt.reshape(n_h, ws, n_w, ws, 3).transpose(0, 2, 1, 3, 4)

        mu_g = gen_b.mean(axis=(2, 3))
        mu_t = tgt_b.mean(axis=(2, 3))

        dg = gen_b - mu_g[:, :, np.newaxis, np.newaxis, :]
        dt = tgt_b - mu_t[:, :, np.newaxis, np.newaxis, :]
        var_g = (dg ** 2).mean(axis=(2, 3))
        var_t = (dt ** 2).mean(axis=(2, 3))
        cov = (dg * dt).mean(axis=(2, 3))

        ssim_map = (
            (2 * mu_g * mu_t + self._C1) * (2 * cov + self._C2)
        ) / (
            (mu_g ** 2 + mu_t ** 2 + self._C1) * (var_g + var_t + self._C2)
        )

        return float(np.clip(ssim_map.mean(), 0.0, 1.0))

    def _ssim_global(self, gen: np.ndarray, tgt: np.ndarray) -> float:
        mu_g = gen.mean(axis=(0, 1))
        mu_t = tgt.mean(axis=(0, 1))
        dg = gen - mu_g
        dt = tgt - mu_t
        var_g = (dg ** 2).mean(axis=(0, 1))
        var_t = (dt ** 2).mean(axis=(0, 1))
        cov = (dg * dt).mean(axis=(0, 1))
        ssim = (
            (2 * mu_g * mu_t + self._C1) * (2 * cov + self._C2)
        ) / (
            (mu_g ** 2 + mu_t ** 2 + self._C1) * (var_g + var_t + self._C2)
        )
        return float(np.clip(ssim.mean(), 0.0, 1.0))
