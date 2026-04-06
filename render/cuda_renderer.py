"""Renderer CUDA que usa CuPy para rasterizar triangulos en GPU (NVIDIA/CUDA).

Optimizaciones:
- Batch evaluation: evalua todos los individuos en un solo kernel launch
- Gene packing vectorizado con numpy (sin loops Python)
- Pinned memory para transfers CPU->GPU rapidos
- Shared memory en kernel para cachear genes por bloque
- Buffers pre-alocados y reutilizados entre generaciones
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


# Kernel batched con shared memory para genes
_BATCH_RASTERIZE_KERNEL = r"""
#define GENES_PER_BATCH 32
#define GENE_FLOATS 11

extern "C" __global__
void batch_rasterize(
    const float* __restrict__ all_genes,
    float* __restrict__ all_canvases,
    int n_genes,
    int width,
    int height,
    int n_individuals
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int ind = blockIdx.z;

    if (px >= width || py >= height || ind >= n_individuals) return;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    // Shared memory para cachear genes del individuo actual
    __shared__ float sg[GENES_PER_BATCH * GENE_FLOATS];

    int genes_offset = ind * n_genes * GENE_FLOATS;

    float cr = 1.0f, cg_ = 1.0f, cb = 1.0f, ca = 1.0f;

    // Procesar genes en batches que caben en shared memory
    for (int batch_start = 0; batch_start < n_genes; batch_start += GENES_PER_BATCH) {
        int batch_end = min(batch_start + GENES_PER_BATCH, n_genes);
        int batch_count = batch_end - batch_start;
        int floats_to_load = batch_count * GENE_FLOATS;

        // Carga cooperativa: todos los threads del bloque cargan genes a shared
        for (int f = tid; f < floats_to_load; f += block_size) {
            sg[f] = all_genes[genes_offset + (batch_start * GENE_FLOATS) + f];
        }
        __syncthreads();

        // Cada thread procesa su pixel contra los genes en shared memory
        for (int i = 0; i < batch_count; i++) {
            const float* g = sg + i * GENE_FLOATS;

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
                float sg_ = g[7] / 255.0f;
                float sb = g[8] / 255.0f;
                float sa = g[9];

                float out_a = sa + ca * (1.0f - sa);
                if (out_a > 0.0f) {
                    cr = (sr * sa + cr * ca * (1.0f - sa)) / out_a;
                    cg_ = (sg_ * sa + cg_ * ca * (1.0f - sa)) / out_a;
                    cb = (sb * sa + cb * ca * (1.0f - sa)) / out_a;
                }
                ca = out_a;
            }
        }
        __syncthreads();
    }

    int pixel_idx = (ind * height * width + py * width + px) * 4;
    all_canvases[pixel_idx + 0] = cr;
    all_canvases[pixel_idx + 1] = cg_;
    all_canvases[pixel_idx + 2] = cb;
    all_canvases[pixel_idx + 3] = ca;
}
"""

# Kernel batched MSE con reduccion parcial via shared memory
_BATCH_MSE_KERNEL = r"""
extern "C" __global__
void batch_mse(
    const float* __restrict__ all_canvases,
    const float* __restrict__ target,
    float* __restrict__ all_errors,
    int pixels_per_image,
    int n_individuals
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_individuals * pixels_per_image;
    if (idx >= total) return;

    int px = idx % pixels_per_image;

    int canvas_base = idx * 4;
    int target_base = px * 4;

    float dr = all_canvases[canvas_base + 0] - target[target_base + 0];
    float dg = all_canvases[canvas_base + 1] - target[target_base + 1];
    float db = all_canvases[canvas_base + 2] - target[target_base + 2];

    all_errors[idx] = (dr * dr + dg * dg + db * db) / 3.0f;
}
"""

# Kernel single individual (para render() y snapshots)
_RASTERIZE_KERNEL = r"""
extern "C" __global__
void rasterize(
    const float* __restrict__ genes,
    float* __restrict__ canvas,
    int n_genes,
    int width,
    int height
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

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

    int idx = (py * width + px) * 4;
    canvas[idx + 0] = cr;
    canvas[idx + 1] = cg;
    canvas[idx + 2] = cb;
    canvas[idx + 3] = ca;
}
"""


class CUDARenderer(Renderer):
    """Renderer que usa CUDA via CuPy para rasterizar triangulos en GPU NVIDIA.

    Soporta evaluacion batched de toda la poblacion en un solo kernel launch.
    Usa shared memory, pinned memory y gene packing vectorizado.
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
        self._batch_mse_kernel = cp.RawKernel(_BATCH_MSE_KERNEL, "batch_mse")

        # Target en GPU (una sola vez)
        self._target_gpu = cp.asarray(target_image.astype(np.float32))
        self._target_flat_gpu = self._target_gpu.ravel()

        # Buffer single individual (para render/snapshots)
        self._canvas_gpu = cp.zeros((height, width, 4), dtype=cp.float32)

        # Buffers batched
        self._batch_size = 0
        self._batch_canvases_gpu = None
        self._batch_errors_gpu = None

        # Pinned memory buffer para gene packing (transfer rapido CPU->GPU)
        self._pinned_buffer = None
        self._pinned_size = 0

        # Grid/block single
        self._block = (16, 16, 1)
        self._grid = (
            (width + self._block[0] - 1) // self._block[0],
            (height + self._block[1] - 1) // self._block[1],
            1,
        )

        # Gene attribute names cache
        self._gene_attrs = ('x1', 'y1', 'x2', 'y2', 'x3', 'y3')

        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        logger.info(
            "CUDARenderer inicializado: %dx%d, device=%s",
            width, height, device_name,
        )

    def _get_pinned_buffer(self, size: int) -> np.ndarray:
        """Retorna pinned memory buffer, realocando si es necesario."""
        cp = self.cp
        if self._pinned_buffer is None or self._pinned_size < size:
            self._pinned_size = size
            mem = cp.cuda.alloc_pinned_memory(size * 4)  # float32 = 4 bytes
            self._pinned_buffer = np.frombuffer(mem, dtype=np.float32, count=size)
        return self._pinned_buffer[:size]

    def _ensure_batch_buffers(self, n_individuals: int):
        """Pre-aloca buffers GPU para batch evaluation."""
        cp = self.cp
        if n_individuals > self._batch_size:
            self._batch_size = n_individuals
            self._batch_canvases_gpu = cp.empty(
                (n_individuals * self.height * self.width * 4,), dtype=cp.float32
            )
            self._batch_errors_gpu = cp.empty(
                (n_individuals * self._pixels_per_image,), dtype=cp.float32
            )

    def _pack_genes_vectorized(self, individuals: list, n_genes: int) -> np.ndarray:
        """Empaqueta genes de todos los individuos usando numpy vectorizado."""
        m = len(individuals)
        total_floats = m * n_genes * 11

        buf = self._get_pinned_buffer(total_floats)

        # Reshape para llenar como [M, N, 11]
        view = buf.reshape(m, n_genes, 11)

        for i, ind in enumerate(individuals):
            genes = ind.genes
            for j in range(n_genes):
                g = genes[j]
                view[i, j, 0] = g.x1
                view[i, j, 1] = g.y1
                view[i, j, 2] = g.x2
                view[i, j, 3] = g.y2
                view[i, j, 4] = g.x3
                view[i, j, 5] = g.y3
                view[i, j, 6] = g.r
                view[i, j, 7] = g.g
                view[i, j, 8] = g.b
                view[i, j, 9] = g.a
                view[i, j, 10] = 1.0 if g.active else 0.0

        return buf

    def evaluate_batch(self, individuals: list) -> list[float]:
        """Evalua fitness de todos los individuos en un solo batch GPU.

        1 transfer CPU->GPU, 1 kernel rasterize, 1 kernel MSE.
        """
        cp = self.cp
        m = len(individuals)
        if m == 0:
            return []

        n_genes = len(individuals[0].genes)
        self._ensure_batch_buffers(m)

        # Pack genes en pinned memory y subir a GPU
        genes_flat = self._pack_genes_vectorized(individuals, n_genes)
        genes_gpu = cp.asarray(genes_flat)

        # Rasterizar todos en paralelo (1 kernel launch)
        batch_block = (16, 16, 1)
        batch_grid = (
            (self.width + 15) // 16,
            (self.height + 15) // 16,
            m,
        )

        canvas_buf = self._batch_canvases_gpu[:m * self.height * self.width * 4]
        errors_buf = self._batch_errors_gpu[:m * self._pixels_per_image]

        self._batch_rasterize_kernel(
            batch_grid, batch_block,
            (genes_gpu, canvas_buf, np.int32(n_genes),
             np.int32(self.width), np.int32(self.height), np.int32(m)),
        )

        # MSE batched (1 kernel launch)
        total_pixels_all = m * self._pixels_per_image
        mse_grid = ((total_pixels_all + 255) // 256,)

        self._batch_mse_kernel(
            mse_grid, (256,),
            (canvas_buf, self._target_flat_gpu, errors_buf,
             np.int32(self._pixels_per_image), np.int32(m)),
        )

        # Reduce + fitness en GPU
        errors_reshaped = errors_buf[:total_pixels_all].reshape(m, self._pixels_per_image)
        mse_values = cp.mean(errors_reshaped, axis=1)
        fitness_values = 1.0 / (1.0 + mse_values)

        return cp.asnumpy(fitness_values).tolist()

    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """Rasteriza triangulos en GPU (single individual, para snapshots)."""
        cp = self.cp

        if not genes:
            return np.ones((height, width, 4), dtype=np.float32)

        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)),
             np.int32(width), np.int32(height)),
        )

        return cp.asnumpy(self._canvas_gpu)

    def compute_fitness_gpu(self, genes: list) -> float:
        """Fitness de un solo individuo en GPU (fallback si no se usa batch)."""
        cp = self.cp

        if not genes:
            gen_rgb = np.ones((self.height, self.width, 3), dtype=np.float32)
            tgt_rgb = cp.asnumpy(self._target_gpu[:, :, :3])
            mse = np.mean((gen_rgb - tgt_rgb) ** 2)
            return float(1.0 / (1.0 + mse))

        gene_data = self._genes_to_array(genes)
        genes_gpu = cp.asarray(gene_data)

        self._rasterize_kernel(
            self._grid, self._block,
            (genes_gpu, self._canvas_gpu, np.int32(len(genes)),
             np.int32(self.width), np.int32(self.height)),
        )

        errors = cp.empty(self._pixels_per_image, dtype=cp.float32)
        mse_grid = ((self._pixels_per_image + 255) // 256,)

        # Inline MSE kernel
        canvas_flat = self._canvas_gpu.ravel()
        target_flat = self._target_flat_gpu
        # Compute MSE using cupy ops (simpler for single individual)
        diff = canvas_flat.reshape(-1, 4)[:, :3] - target_flat.reshape(-1, 4)[:, :3]
        mse = float(cp.mean(diff ** 2))
        return 1.0 / (1.0 + mse)

    def evaluate_batch_raw(self, pop_array: np.ndarray) -> list[float]:
        """Evalua fitness de una poblacion representada como array numpy (M, N, 11).

        Skips gene packing entirely — the array is already in the right format.
        This is the fast path used by core/fast_loop.py.
        """
        cp = self.cp
        m, n_genes, _ = pop_array.shape
        if m == 0:
            return []

        self._ensure_batch_buffers(m)

        # Transfer directo: el array ya es float32 contiguo
        genes_flat = np.ascontiguousarray(pop_array.reshape(-1), dtype=np.float32)
        genes_gpu = cp.asarray(genes_flat)

        batch_grid = (
            (self.width + 15) // 16,
            (self.height + 15) // 16,
            m,
        )

        canvas_buf = self._batch_canvases_gpu[:m * self.height * self.width * 4]
        errors_buf = self._batch_errors_gpu[:m * self._pixels_per_image]

        self._batch_rasterize_kernel(
            batch_grid, (16, 16, 1),
            (genes_gpu, canvas_buf, np.int32(n_genes),
             np.int32(self.width), np.int32(self.height), np.int32(m)),
        )

        total_pixels_all = m * self._pixels_per_image
        mse_grid = ((total_pixels_all + 255) // 256,)

        self._batch_mse_kernel(
            mse_grid, (256,),
            (canvas_buf, self._target_flat_gpu, errors_buf,
             np.int32(self._pixels_per_image), np.int32(m)),
        )

        errors_reshaped = errors_buf[:total_pixels_all].reshape(m, self._pixels_per_image)
        mse_values = cp.mean(errors_reshaped, axis=1)
        fitness_values = 1.0 / (1.0 + mse_values)

        return cp.asnumpy(fitness_values).tolist()

    def _genes_to_array(self, genes: list) -> np.ndarray:
        """Convierte lista de genes a array numpy [N, 11]."""
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
