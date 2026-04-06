import numpy as np
from fitness.fitness_function import FitnessFunction


class MSSSIMFitness(FitnessFunction):
    """
    MS-SSIM: Multi-Scale Structural Similarity.

    Evalua SSIM en 5 escalas de resolucion con pesos perceptuales.
    Mejor correlacion con juicios humanos que SSIM de escala unica,
    especialmente para distorsiones de blur y compresion.

    Referencia: Wang, Simoncelli, Bovik, Asilomar 2003.
    """

    name = "msssim"

    _WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)
    _C1 = 0.01 ** 2
    _C2 = 0.03 ** 2
    _N_SCALES = 5
    _BLOCK_SIZE = 8

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        gen = generated[:, :, :3].astype(np.float64)
        tgt = target[:, :, :3].astype(np.float64)

        cs_per_scale = []
        luminance_last = None

        for scale in range(self._N_SCALES):
            l, cs = self._ssim_components(gen, tgt)
            cs_per_scale.append(cs)
            if scale == self._N_SCALES - 1:
                luminance_last = l
            # Downsample para la siguiente escala
            gen = self._downsample(gen)
            tgt = self._downsample(tgt)

        # MS-SSIM como producto ponderado
        ms_ssim = luminance_last ** self._WEIGHTS[-1]
        for j, cs in enumerate(cs_per_scale):
            ms_ssim *= cs ** self._WEIGHTS[j]

        return float(np.clip(ms_ssim, 0.0, 1.0))

    def _ssim_components(self, gen: np.ndarray, tgt: np.ndarray):
        """Calcula luminancia y contraste-estructura globales o por bloques."""
        H, W = gen.shape[:2]
        ws = self._BLOCK_SIZE

        if H < ws or W < ws:
            # Estadisticas globales si la imagen es muy chica
            mu_g = gen.mean(axis=(0, 1))
            mu_t = tgt.mean(axis=(0, 1))
            dg = gen - mu_g
            dt = tgt - mu_t
            var_g = (dg ** 2).mean(axis=(0, 1))
            var_t = (dt ** 2).mean(axis=(0, 1))
            cov = (dg * dt).mean(axis=(0, 1))
        else:
            H_c = (H // ws) * ws
            W_c = (W // ws) * ws
            g = gen[:H_c, :W_c]
            t = tgt[:H_c, :W_c]
            n_h, n_w = H_c // ws, W_c // ws

            g_b = g.reshape(n_h, ws, n_w, ws, 3).transpose(0, 2, 1, 3, 4)
            t_b = t.reshape(n_h, ws, n_w, ws, 3).transpose(0, 2, 1, 3, 4)

            mu_g = g_b.mean(axis=(2, 3))
            mu_t = t_b.mean(axis=(2, 3))
            dg = g_b - mu_g[:, :, np.newaxis, np.newaxis, :]
            dt = t_b - mu_t[:, :, np.newaxis, np.newaxis, :]
            var_g = (dg ** 2).mean(axis=(2, 3))
            var_t = (dt ** 2).mean(axis=(2, 3))
            cov = (dg * dt).mean(axis=(2, 3))

        luminance = float(np.mean(
            (2 * mu_g * mu_t + self._C1) / (mu_g ** 2 + mu_t ** 2 + self._C1)
        ))
        cs = float(np.mean(
            (2 * cov + self._C2) / (var_g + var_t + self._C2)
        ))
        return np.clip(luminance, 0.0, 1.0), np.clip(cs, 0.0, 1.0)

    def _downsample(self, img: np.ndarray) -> np.ndarray:
        """Reduce imagen a la mitad promediando bloques 2x2."""
        H, W = img.shape[:2]
        H2, W2 = H // 2, W // 2
        if H2 == 0 or W2 == 0:
            return img
        return img[:H2 * 2, :W2 * 2].reshape(H2, 2, W2, 2, -1).mean(axis=(1, 3))
