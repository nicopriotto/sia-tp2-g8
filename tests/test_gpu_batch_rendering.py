import pytest
import numpy as np

from render.gpu_renderer import GPURenderer, gpu_available
from genes.triangle_gene import TriangleGene

skip_no_gpu = pytest.mark.skipif(not gpu_available(), reason="GPU no disponible")


@skip_no_gpu
def test_render_batch_produce_imagen_correcta():
    w, h = 64, 64
    target = np.ones((h, w, 4), dtype=np.float32)
    genes = np.array([
        TriangleGene(x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0, r=255, g=0, b=0, a=1.0).to_row(),
        TriangleGene(x1=0.0, y1=0.0, x2=0.5, y2=0.5, x3=0.0, y3=1.0, r=0, g=255, b=0, a=0.8).to_row(),
        TriangleGene(x1=0.3, y1=0.3, x2=0.7, y2=0.3, x3=0.5, y3=0.7, r=0, g=0, b=255, a=0.6).to_row(),
        TriangleGene(x1=0.5, y1=0.0, x2=1.0, y2=0.5, x3=0.5, y3=0.5, r=128, g=128, b=0, a=0.5).to_row(),
        TriangleGene(x1=0.0, y1=0.5, x2=1.0, y2=0.5, x3=0.5, y3=1.0, r=0, g=128, b=128, a=0.4).to_row(),
    ])
    gpu = GPURenderer(w, h, target)
    try:
        result = gpu.render(genes, w, h)
        assert result.shape == (h, w, 4)
        assert result.dtype == np.float32
        assert not np.allclose(result[:, :, :3], 1.0)
    finally:
        gpu.release()


@skip_no_gpu
def test_render_sin_genes_activos_retorna_blanco():
    w, h = 64, 64
    target = np.ones((h, w, 4), dtype=np.float32)
    genes = np.array([
        TriangleGene(x1=0.1, y1=0.1, x2=0.5, y2=0.1, x3=0.3, y3=0.5, r=255, g=0, b=0, a=0.8, active=False).to_row(),
        TriangleGene(x1=0.4, y1=0.3, x2=0.9, y2=0.3, x3=0.6, y3=0.8, r=0, g=255, b=0, a=0.6, active=False).to_row(),
    ])
    gpu = GPURenderer(w, h, target)
    try:
        result = gpu.render(genes, w, h)
        assert np.allclose(result[:, :, :3], 1.0)
    finally:
        gpu.release()


@skip_no_gpu
def test_render_lista_vacia_no_falla():
    w, h = 64, 64
    target = np.ones((h, w, 4), dtype=np.float32)
    gpu = GPURenderer(w, h, target)
    try:
        genes = np.empty((0, 11), dtype=np.float64)
        result = gpu.render(genes, w, h)
        assert result.shape == (h, w, 4)
        assert np.allclose(result[:, :, :3], 1.0)
    finally:
        gpu.release()
