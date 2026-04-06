import numpy as np
from fitness.fitness_function import FitnessFunction


class FSIMFitness(FitnessFunction):
    """
    FSIM: Feature Similarity Index.

    Usa Phase Congruency (PC) y Gradient Magnitude (GM) como features
    del sistema visual humano. PC detecta bordes independientemente
    del contraste, haciendo FSIM el indice mas biologicamente fundamentado.

    Referencia: Zhang et al., IEEE TIP 2011.
    """

    name = "fsim"

    _N_SCALES = 4
    _N_ORIENTATIONS = 4
    _MIN_WAVELENGTH = 6.0
    _MULT = 2.0
    _SIGMA_ONF = 0.55
    _SIGMA_THETA = np.pi / 6
    _NOISE_FACTOR = 5.0

    _T1 = 0.85
    _T2 = (160.0 / 255.0) ** 2
    _T3 = (200.0 / 255.0) ** 2

    _Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 3.0
    _Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 3.0

    def compute(self, generated: np.ndarray, target: np.ndarray) -> float:
        gen_y = self._to_luminance(generated[:, :, :3])
        tgt_y = self._to_luminance(target[:, :, :3])

        pc_gen = self._phase_congruency(gen_y)
        pc_tgt = self._phase_congruency(tgt_y)
        gm_gen = self._gradient_magnitude(gen_y)
        gm_tgt = self._gradient_magnitude(tgt_y)

        s_pc = (2.0 * pc_gen * pc_tgt + self._T1) / (pc_gen ** 2 + pc_tgt ** 2 + self._T1)
        s_gm = (2.0 * gm_gen * gm_tgt + self._T2) / (gm_gen ** 2 + gm_tgt ** 2 + self._T2)
        s_l = s_pc * s_gm

        pc_m = np.maximum(pc_gen, pc_tgt)
        weight_sum = pc_m.sum()

        if weight_sum < 1e-8:
            # Sin estructura detectable (imagenes uniformes o identicas):
            # usar media no ponderada de s_l como fallback
            fsim = float(np.mean(s_l))
        else:
            fsim = float((s_l * pc_m).sum() / weight_sum)
        return float(np.clip(fsim, 0.0, 1.0))

    def _to_luminance(self, rgb: np.ndarray) -> np.ndarray:
        return (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float64)

    def _phase_congruency(self, gray: np.ndarray) -> np.ndarray:
        H, W = gray.shape
        fy = np.fft.fftfreq(H)[:, np.newaxis]
        fx = np.fft.fftfreq(W)[np.newaxis, :]
        freq_r = np.sqrt(fx ** 2 + fy ** 2)
        freq_r[0, 0] = 1.0
        theta_grid = np.arctan2(fy, fx)

        img_fft = np.fft.fft2(gray)

        energy = np.zeros((H, W), dtype=np.float64)
        sum_ampl = np.zeros((H, W), dtype=np.float64)
        noise_threshold = 0.0

        for s in range(self._N_SCALES):
            wavelength = self._MIN_WAVELENGTH * (self._MULT ** s)
            f_0 = 1.0 / wavelength

            lg = np.exp(-(np.log(freq_r / f_0)) ** 2 / (2.0 * self._SIGMA_ONF ** 2))
            lg[0, 0] = 0.0

            for o in range(self._N_ORIENTATIONS):
                theta_o = o * np.pi / self._N_ORIENTATIONS
                ds = np.sin(theta_grid) * np.cos(theta_o) - np.cos(theta_grid) * np.sin(theta_o)
                dc = np.cos(theta_grid) * np.cos(theta_o) + np.sin(theta_grid) * np.sin(theta_o)
                dtheta = np.abs(np.arctan2(ds, dc))
                angular = np.exp(-dtheta ** 2 / (2.0 * self._SIGMA_THETA ** 2))

                filtro = lg * angular
                respuesta = np.fft.ifft2(img_fft * filtro)
                re = respuesta.real
                im = respuesta.imag
                amplitud = np.sqrt(re ** 2 + im ** 2)

                energy += re
                sum_ampl += amplitud

                if s == 0 and o == 0:
                    tau = np.median(amplitud) / np.sqrt(np.log(4))
                    noise_threshold = tau * self._NOISE_FACTOR * np.sqrt(2.0 * np.log(H * W))

        pc = np.maximum(0.0, (np.abs(energy) - noise_threshold)) / (sum_ampl + 1e-8)
        return pc

    def _conv2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        padded = np.pad(img, 1, mode='reflect')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        return (windows * kernel).sum(axis=(-1, -2))

    def _gradient_magnitude(self, gray: np.ndarray) -> np.ndarray:
        gray32 = gray.astype(np.float32)
        gx = self._conv2d(gray32, self._Kx)
        gy = self._conv2d(gray32, self._Ky)
        return np.sqrt(gx ** 2 + gy ** 2).astype(np.float64)
