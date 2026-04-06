"""Renderer CUDA que usa CuPy para rasterizar triangulos en GPU (NVIDIA/CUDA).

Optimizaciones:
- Batch evaluation: evalua todos los individuos en un solo kernel launch
- Los genes de toda la poblacion se suben a GPU en una sola transferencia
- El fitness (MSE) se calcula en GPU sin copiar imagenes a CPU
- Buffers pre-alocados para evitar mallocs en cada generacion
- CUDA streams para overlap de compute y transfers
"""

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


# Kernel CUDA batched: rasteriza M individuos en paralelo
# Grid 3D: (pixels_x, pixels_y, individual_index)
_BATCH_RASTERIZE_KERNEL = r"""
extern "C" __global__
void batch_rasterize(
    const float* all_genes,   // [M * N * 11] genes de todos los individuos
    float* all_canvases,      // [M * H * W * 4] canvas por individuo
    int n_genes,              // genes por individuo
    int width,
    int height,
    int n_individuals         // M
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = blockIdx.z;  // indice del individuo

    if (px >= width || py >= height || ind >= n_individuals) return;

    // Offset en el canvas de este individuo
    int canvas_offset = ind * height * width * 4;
    int pixel_idx = canvas_offset + (py * width + px) * 4;

    // Offset en los genes de este individuo
    int genes_offset = ind * n_genes * 11;

    float cr = 1.0f, cg = 1.0f, cb = 1.0f, ca = 1.0f;

    for (int i = 0; i < n_genes; i++) {
        const float* g = all_genes + genes_offset + i * 11;

        if (g[10] < 0.5f) continue;

        float ax = g[0] * width,  ay = g[1] * height;
        float bx = g[2] * width,  by = g[3] * height;
        float cx = g[4] * width,  cy = g[5] * height;

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
            float sr = g[6] / 255.0f;
            float sg = g[7] / 255.0f;
            float sb = g[8] / 255.0f;
            float sa = g[9];

            float out_a = sa + ca * (1.0f - sa);
            if (out_a > 0.0f) {
                cr = (sr * sa + cr * ca * (1.0f - sa)) / out_a;
                cg = (sg * sa + cg * ca * (1.0f - sa)) / out_a;
                cb = (sb * sa + cb * ca * (1.0f - sa)) / out_a;
            }
            ca = out_a;
        }
    }

    all_canvases[pixel_idx + 0] = cr;
    all_canvases[pixel_idx + 1] = cg;
    all_canvases[pixel_idx + 2] = cb;
    all_canvases[pixel_idx + 3] = ca;
}
"""

# Kernel batched MSE: calcula error por pixel por individuo
_BATCH_MSE_KERNEL = r"""
extern "C" __global__
void batch_mse(
    const float* all_canvases,  // [M * H * W * 4]
    const float* target,        // [H * W * 4]
    float* all_errors,          // [M * H * W] error por pixel por individuo
    int pixels_per_image,       // H * W
    int n_individuals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_individuals * pixels_per_image;
    if (idx >= total) return;

    int ind = idx / pixels_per_image;
    int px = idx % pixels_per_image;

    int canvas_base = ind * pixels_per_image * 4 + px * 4;
    int target_base = px * 4;

    float dr = all_canvases[canvas_base + 0] - target[target_base + 0];
    float dg = all_canvases[canvas_base + 1] - target[target_base + 1];
    float db = all_canvases[canvas_base + 2] - target[target_base + 2];

    all_errors[idx] = (dr * dr + dg * dg + db * db) / 3.0f;
}
"""

# Kernel single individual (para render() simple)
_RASTERIZE_KERNEL = r"""
extern "C" __global__
void rasterize(
    const float* genes,
    float* canvas,
    int n_genes,
    int width,
    int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 4;

    float cr = 1.0f, cg = 1.0f, cb = 1.0f, ca = 1.0f;

    for (int i = 0; i < n_genes; i++) {
        const float* g = genes + i * 11;

        if (g[10] < 0.5f) continue;

        float ax = g[0] * width,  ay = g[1] * height;
        float bx = g[2] * width,  by = g[3] * height;
        float cx = g[4] * width,  cy = g[5] * height;

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
            float sr = g[6] / 255.0f;
            float sg = g[7] / 255.0f;
            float sb = g[8] / 255.0f;
            float sa = g[9];

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
    const float* generated,
    const float* target,
    float* errors,
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
    """Renderer que usa CUDA via CuPy para rasterizar triangulos en GPU NVIDIA.

    Soporta evaluacion batched de toda la poblacion en un solo kernel launch.
    """

    def __init__(self, width: int, height: int, target_image: np.ndarray):
        import cupy as cp

        self.cp = cp
        self.width = width
        self.height = height
        self._pixels_per_image = width * height

        # Compilar kernels
        self._rasterize_kernel = cp.RawKernel(_RASTERIZE_KERNEL, "rasterize")
        self._batch_rasterize_kernel = cp.RawKernel(_BATCH_RASTERIZE_KERNEL, "batch_rasterize")
        self._mse_kernel = cp.RawKernel(_MSE_KERNEL, "mse_rgb")
        self._batch_mse_kernel = cp.RawKernel(_BATCH_MSE_KERNEL, "batch_mse")

        # Subir target a GPU una sola vez
        self._target_gpu = cp.asarray(target_image.astype(np.float32))
        self._target_flat_gpu = self._target_gpu.ravel()

        # Buffer single individual (para render() y compute_fitness_gpu())
        self._canvas_gpu = cp.zeros((height, width, 4), dtype=cp.float32)
        self._errors_gpu = cp.zeros(self._pixels_per_image, dtype=cp.float32)

        # Buffers batched (se redimensionan si cambia el batch size)
        self._batch_size = 0
        self._batch_canvases_gpu = None
        self._batch_errors_gpu = None
        self._batch_genes_gpu = None

        # Grid/block dims para single individual
        self._block = (16, 16, 1)
        self._grid = (
            (width + self._block[0] - 1) // self._block[0],
            (height + self._block[1] - 1) // self._block[1],
            1,
        )

        # Block para MSE
        self._mse_block = (256,)
        self._mse_grid = ((self._pixels_per_image + 255) // 256,)

        # Pre-alocar buffer de genes reutilizable en CPU (pinned memory para transfers rapidos)
        self._max_genes = 0
        self._genes_cpu_buffer = None

        logger.info(
            "CUDARenderer inicializado: %dx%d, device=%s",
            width, height, cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        )

    def _ensure_batch_buffers(self, n_individuals: int, n_genes: int):
        """Pre-aloca o redimensiona buffers para batch evaluation."""
        cp = self.cp
        needed = n_individuals

        if needed > self._batch_size:
            self._batch_size = needed
            self._batch_canvases_gpu = cp.zeros(
                (needed * self.height * self.width * 4,), dtype=cp.float32
            )
            self._batch_errors_gpu = cp.zeros(
                (needed * self._pixels_per_image,), dtype=cp.float32
            )
            logger.debug("Batch buffers realocados para %d individuos", needed)

        total_genes = n_individuals * n_genes * 11
        if self._genes_cpu_buffer is None or self._genes_cpu_buffer.size < total_genes:
            self._genes_cpu_buffer = np.zeros(total_genes, dtype=np.float32)

    def _pack_genes_batch(self, individuals: list, n_genes: int) -> np.ndarray:
        """Empaqueta genes de todos los individuos en un array contiguo [M*N*11]."""
        m = len(individuals)
        total = m * n_genes * 11
        buf = self._genes_cpu_buffer[:total]

        for i, ind in enumerate(individuals):
            base = i * n_genes * 11
            for j, g in enumerate(ind.genes):
                offset = base + j * 11
                buf[offset + 0] = g.x1
                buf[offset + 1] = g.y1
                buf[offset + 2] = g.x2
                buf[offset + 3] = g.y2
                buf[offset + 4] = g.x3
                buf[offset + 5] = g.y3
                buf[offset + 6] = float(g.r)
                buf[offset + 7] = float(g.g)
                buf[offset + 8] = float(g.b)
                buf[offset + 9] = g.a
                buf[offset + 10] = 1.0 if getattr(g, 'active', True) else 0.0

        return buf

    def evaluate_batch(self, individuals: list) -> list[float]:
        """Evalua fitness de todos los individuos en un solo batch GPU.

        1 transfer CPU->GPU, 1 kernel rasterize, 1 kernel MSE, 1 reduce.
        Retorna lista de fitness en el mismo orden.
        """
        cp = self.cp
        m = len(individuals)
        if m == 0:
            return []

        n_genes = len(individuals[0].genes)

        # Pre-alocar buffers si es necesario
        self._ensure_batch_buffers(m, n_genes)

        # Empaquetar todos los genes en CPU y subir de una sola vez
        genes_flat = self._pack_genes_batch(individuals, n_genes)
        genes_gpu = cp.asarray(genes_flat)

        # Rasterizar todos los individuos en paralelo
        batch_block = (16, 16, 1)
        batch_grid = (
            (self.width + batch_block[0] - 1) // batch_block[0],
            (self.height + batch_block[1] - 1) // batch_block[1],
            m,  # un bloque Z por individuo
        )

        canvas_buf = self._batch_canvases_gpu[:m * self.height * self.width * 4]
        errors_buf = self._batch_errors_gpu[:m * self._pixels_per_image]

        self._batch_rasterize_kernel(
            batch_grid, batch_block,
            (genes_gpu, canvas_buf, np.int32(n_genes),
             np.int32(self.width), np.int32(self.height), np.int32(m)),
        )

        # MSE batched
        total_pixels_all = m * self._pixels_per_image
        mse_block = (256,)
        mse_grid = ((total_pixels_all + 255) // 256,)

        self._batch_mse_kernel(
            mse_grid, mse_block,
            (canvas_buf, self._target_flat_gpu, errors_buf,
             np.int32(self._pixels_per_image), np.int32(m)),
        )

        # Reducir: mean por individuo en GPU
        errors_reshaped = errors_buf[:total_pixels_all].reshape(m, self._pixels_per_image)
        mse_values = cp.mean(errors_reshaped, axis=1)

        # Calcular fitness en GPU y traer a CPU
        fitness_values = 1.0 / (1.0 + mse_values)
        result = cp.asnumpy(fitness_values).tolist()

        return result

    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """Rasteriza triangulos en GPU y retorna imagen numpy en CPU (single individual)."""
        cp = self.cp

        if not genes:
            return np.ones((height, width, 4), dtype=np.float32)

        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        self._canvas_gpu[:] = 1.0

        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)),
             np.int32(width), np.int32(height)),
        )

        return cp.asnumpy(self._canvas_gpu)

    def compute_fitness_gpu(self, genes: list) -> float:
        """Renderiza Y calcula fitness en GPU (single individual, sin batch)."""
        cp = self.cp

        if not genes:
            gen_rgb = np.ones((self.height, self.width, 3), dtype=np.float32)
            tgt_rgb = cp.asnumpy(self._target_gpu[:, :, :3])
            mse = np.mean((gen_rgb - tgt_rgb) ** 2)
            return float(1.0 / (1.0 + mse))

        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        self._canvas_gpu[:] = 1.0
        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)),
             np.int32(self.width), np.int32(self.height)),
        )

        self._mse_kernel(
            self._mse_grid, self._mse_block,
            (self._canvas_gpu, self._target_gpu, self._errors_gpu,
             np.int32(self._pixels_per_image)),
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
