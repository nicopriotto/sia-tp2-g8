"""Renderer CUDA que usa CuPy para rasterizar triangulos en GPU (NVIDIA/CUDA)."""

import logging
import numpy as np
from render.renderer import Renderer

logger = logging.getLogger(__name__)


def cuda_available() -> bool:
    """Retorna True si CuPy esta instalado y puede acceder a una GPU CUDA."""
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


# Kernel CUDA: por cada pixel, itera sobre los triangulos y aplica alpha compositing
_RASTERIZE_KERNEL = r"""
extern "C" __global__
void rasterize(
    const float* genes,   // [N, 10]: x1,y1,x2,y2,x3,y3, r,g,b, a
    float* canvas,        // [H, W, 4]: RGBA float32 en [0,1]
    int n_genes,
    int width,
    int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 4;

    // Canvas empieza blanco opaco
    float cr = 1.0f, cg = 1.0f, cb = 1.0f, ca = 1.0f;

    for (int i = 0; i < n_genes; i++) {
        const float* g = genes + i * 11;

        // Chequear si el gen esta activo (indice 10)
        if (g[10] < 0.5f) continue;

        // Coordenadas del triangulo en pixeles
        float ax = g[0] * width,  ay = g[1] * height;
        float bx = g[2] * width,  by = g[3] * height;
        float cx = g[4] * width,  cy = g[5] * height;

        // Test punto-en-triangulo usando coordenadas baricentricas
        float v0x = cx - ax, v0y = cy - ay;
        float v1x = bx - ax, v1y = by - ay;
        float v2x = (float)px - ax, v2y = (float)py - ay;

        float dot00 = v0x * v0x + v0y * v0y;
        float dot01 = v0x * v1x + v0y * v1y;
        float dot02 = v0x * v2x + v0y * v2y;
        float dot11 = v1x * v1x + v1y * v1y;
        float dot12 = v1x * v2x + v1y * v2y;

        float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01 + 1e-10f);
        float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        if (u >= 0.0f && v >= 0.0f && (u + v) <= 1.0f) {
            // Pixel esta dentro del triangulo: alpha compositing
            float sr = g[6] / 255.0f;
            float sg = g[7] / 255.0f;
            float sb = g[8] / 255.0f;
            float sa = g[9];

            // Porter-Duff source over
            float out_a = sa + ca * (1.0f - sa);
            if (out_a > 0.0f) {
                cr = (sr * sa + cr * ca * (1.0f - sa)) / out_a;
                cg = (sg * sa + cg * ca * (1.0f - sa)) / out_a;
                cb = (sb * sa + cb * ca * (1.0f - sa)) / out_a;
            }
            ca = out_a;
        }
    }

    canvas[idx + 0] = cr;
    canvas[idx + 1] = cg;
    canvas[idx + 2] = cb;
    canvas[idx + 3] = ca;
}
"""

_MSE_KERNEL = r"""
extern "C" __global__
void mse_rgb(
    const float* generated,  // [H, W, 4]
    const float* target,     // [H, W, 4]
    float* errors,           // [H * W] - error por pixel
    int total_pixels
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pixels) return;

    int base = i * 4;
    float dr = generated[base + 0] - target[base + 0];
    float dg = generated[base + 1] - target[base + 1];
    float db = generated[base + 2] - target[base + 2];

    errors[i] = (dr * dr + dg * dg + db * db) / 3.0f;
}
"""


class CUDARenderer(Renderer):
    """Renderer que usa CUDA via CuPy para rasterizar triangulos en GPU NVIDIA."""

    def __init__(self, width: int, height: int, target_image: np.ndarray):
        import cupy as cp

        self.cp = cp
        self.width = width
        self.height = height

        # Compilar kernels
        self._rasterize_kernel = cp.RawKernel(_RASTERIZE_KERNEL, "rasterize")
        self._mse_kernel = cp.RawKernel(_MSE_KERNEL, "mse_rgb")

        # Subir target a GPU una sola vez
        self._target_gpu = cp.asarray(target_image.astype(np.float32))

        # Pre-allocar buffers
        self._canvas_gpu = cp.zeros((height, width, 4), dtype=cp.float32)
        self._errors_gpu = cp.zeros(height * width, dtype=cp.float32)

        # Calcular grid/block dims para rasterizacion
        self._block = (16, 16, 1)
        self._grid = (
            (width + self._block[0] - 1) // self._block[0],
            (height + self._block[1] - 1) // self._block[1],
            1,
        )

        # Grid/block para MSE
        total_px = height * width
        self._mse_block = (256,)
        self._mse_grid = ((total_px + 255) // 256,)

        logger.info(
            "CUDARenderer inicializado: %dx%d, device=%s",
            width, height, cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        )

    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """Rasteriza triangulos en GPU y retorna imagen numpy en CPU."""
        cp = self.cp

        if not genes:
            return np.ones((height, width, 4), dtype=np.float32)

        # Preparar datos de genes como array [N, 11]
        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        # Limpiar canvas (blanco opaco)
        self._canvas_gpu[:] = 1.0

        # Lanzar kernel
        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)), np.int32(width), np.int32(height)),
        )

        # Copiar resultado a CPU
        result = cp.asnumpy(self._canvas_gpu)
        return result

    def compute_fitness_gpu(self, genes: list) -> float:
        """Renderiza Y calcula fitness en GPU sin copiar a CPU (mas rapido)."""
        cp = self.cp

        if not genes:
            # Canvas blanco vs target
            gen_rgb = np.ones((self.height, self.width, 3), dtype=np.float32)
            tgt_rgb = cp.asnumpy(self._target_gpu[:, :, :3])
            mse = np.mean((gen_rgb - tgt_rgb) ** 2)
            return float(1.0 / (1.0 + mse))

        # Preparar genes
        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        # Rasterizar
        self._canvas_gpu[:] = 1.0
        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)),
             np.int32(self.width), np.int32(self.height)),
        )

        # MSE en GPU
        total_px = self.height * self.width
        self._mse_kernel(
            self._mse_grid, self._mse_block,
            (self._canvas_gpu, self._target_gpu, self._errors_gpu, np.int32(total_px)),
        )

        mse = float(cp.mean(self._errors_gpu))
        return 1.0 / (1.0 + mse)

    def _genes_to_array(self, genes: list) -> np.ndarray:
        """Convierte lista de genes a array numpy [N, 11] para el kernel."""
        n = len(genes)
        data = np.zeros((n, 11), dtype=np.float32)
        for i, g in enumerate(genes):
            data[i, 0] = g.x1
            data[i, 1] = g.y1
            data[i, 2] = g.x2
            data[i, 3] = g.y2
            data[i, 4] = g.x3
            data[i, 5] = g.y3
            data[i, 6] = float(g.r)
            data[i, 7] = float(g.g)
            data[i, 8] = float(g.b)
            data[i, 9] = g.a
            data[i, 10] = 1.0 if getattr(g, 'active', True) else 0.0
        return data
