import numpy as np
from genes.triangle_gene import TriangleGene
from genes import gene_layout
from render.cpu_renderer import CPURenderer


def test_output_shape():
    renderer = CPURenderer()
    genes = gene_layout.random_genes("triangle", 1)
    result = renderer.render(genes, 100, 80)
    assert result.shape == (80, 100, 4)


def test_output_dtype():
    renderer = CPURenderer()
    genes = gene_layout.random_genes("triangle", 1)
    result = renderer.render(genes, 50, 50)
    assert result.dtype == np.float32


def test_output_range():
    renderer = CPURenderer()
    genes = gene_layout.random_genes("triangle", 5)
    result = renderer.render(genes, 60, 60)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_white_canvas_no_genes():
    renderer = CPURenderer()
    genes = np.empty((0, 11), dtype=np.float64)
    result = renderer.render(genes, 50, 50)
    assert np.allclose(result, 1.0)


def test_opaque_red_triangle():
    renderer = CPURenderer()
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=1.0,
    )
    genes = gene.to_row().reshape(1, -1)
    result = renderer.render(genes, 100, 100)
    red_pixels = (result[:, :, 0] > 0.9) & (result[:, :, 1] < 0.1) & (result[:, :, 2] < 0.1)
    assert np.sum(red_pixels) > 0


def test_alpha_zero_transparent():
    renderer = CPURenderer()
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=1.0, y2=0.0, x3=0.5, y3=1.0,
        r=255, g=0, b=0, a=0.0,
    )
    genes = gene.to_row().reshape(1, -1)
    result = renderer.render(genes, 50, 50)
    white_canvas = np.ones((50, 50, 4), dtype=np.float32)
    assert np.allclose(result, white_canvas)
