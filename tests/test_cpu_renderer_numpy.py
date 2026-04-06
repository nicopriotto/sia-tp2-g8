import threading

import numpy as np
import pytest

from render.cpu_renderer import CPURenderer
from genes.triangle_gene import TriangleGene
from genes import gene_layout


def test_canvas_blanco_sin_genes():
    renderer = CPURenderer()
    genes = np.empty((0, 11), dtype=np.float64)
    result = renderer.render(genes, 64, 64)
    assert result.shape == (64, 64, 4)
    assert result.dtype == np.float32
    assert np.allclose(result, 1.0)


def test_triangulo_unico_colorea_pixeles():
    renderer = CPURenderer()
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=1.0,
    )
    genes = gene.to_row().reshape(1, -1)
    result = renderer.render(genes, 64, 64)
    assert result.shape == (64, 64, 4)
    assert not np.allclose(result[:, :, :3], 1.0)


def test_gen_inactivo_no_se_dibuja():
    renderer = CPURenderer()
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=1.0, active=False,
    )
    genes = gene.to_row().reshape(1, -1)
    result = renderer.render(genes, 64, 64)
    assert np.allclose(result[:, :, :3], 1.0)


def test_thread_safety():
    renderer = CPURenderer()
    errors = []

    def worker(idx):
        try:
            genes = gene_layout.random_genes("triangle", 1)
            result = renderer.render(genes, 32, 32)
            assert result.shape == (32, 32, 4)
            assert result.dtype == np.float32
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errores en threads: {errors}"


def test_valores_en_rango():
    renderer = CPURenderer()
    genes = np.array([
        TriangleGene(x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0, r=255, g=0, b=0, a=0.8).to_row(),
        TriangleGene(x1=0.2, y1=0.2, x2=0.8, y2=0.2, x3=0.5, y3=0.8, r=0, g=255, b=0, a=0.6).to_row(),
    ])
    result = renderer.render(genes, 64, 64)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
