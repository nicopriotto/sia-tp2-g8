import pytest
import numpy as np

from render.gpu_renderer import GPURenderer, gpu_available

skip_no_gpu = pytest.mark.skipif(not gpu_available(), reason="GPU no disponible")


def _fixed_genes():
    """5 TriangleGene con valores fijos para tests reproducibles."""
    from genes.triangle_gene import TriangleGene
    return [
        TriangleGene(x1=0.1, y1=0.1, x2=0.5, y2=0.1, x3=0.3, y3=0.5, r=255, g=0, b=0, a=0.8),
        TriangleGene(x1=0.4, y1=0.3, x2=0.9, y2=0.3, x3=0.6, y3=0.8, r=0, g=255, b=0, a=0.6),
        TriangleGene(x1=0.0, y1=0.5, x2=0.5, y2=0.5, x3=0.25, y3=1.0, r=0, g=0, b=255, a=0.5),
        TriangleGene(x1=0.2, y1=0.0, x2=0.8, y2=0.0, x3=0.5, y3=0.4, r=128, g=128, b=0, a=0.7),
        TriangleGene(x1=0.6, y1=0.6, x2=1.0, y2=0.6, x3=0.8, y3=1.0, r=0, g=128, b=128, a=0.4),
    ]


@skip_no_gpu
def test_gpu_output_shape():
    w, h = 50, 50
    target = np.ones((h, w, 4), dtype=np.float32)
    gpu = GPURenderer(w, h, target)
    try:
        result = gpu.render(_fixed_genes(), w, h)
        assert result.shape == (h, w, 4)
        assert result.dtype == np.float32
    finally:
        gpu.release()


@skip_no_gpu
def test_gpu_output_matches_cpu():
    from render.cpu_renderer import CPURenderer
    w, h = 50, 50
    target = np.ones((h, w, 4), dtype=np.float32)
    genes = _fixed_genes()

    cpu = CPURenderer()
    cpu_image = cpu.render(genes, w, h)

    gpu = GPURenderer(w, h, target)
    try:
        gpu_image = gpu.render(genes, w, h)
        # La rasterizacion difiere en bordes entre Pillow y OpenGL,
        # por eso comparamos error medio absoluto en vez de per-pixel.
        mean_diff = float(np.mean(np.abs(gpu_image - cpu_image)))
        assert mean_diff < 0.05, f"Mean diff: {mean_diff:.6f}"
    finally:
        gpu.release()


@skip_no_gpu
def test_gpu_fitness_matches_cpu():
    from render.cpu_renderer import CPURenderer
    from fitness.mse import MSEFitness
    w, h = 50, 50
    target = np.random.rand(h, w, 4).astype(np.float32)
    genes = _fixed_genes()

    cpu = CPURenderer()
    cpu_image = cpu.render(genes, w, h)
    cpu_fitness = MSEFitness().compute(cpu_image, target)

    gpu = GPURenderer(w, h, target)
    try:
        gpu_fitness = gpu.compute_fitness(genes, "mse")
        assert abs(gpu_fitness - cpu_fitness) < 0.02, (
            f"GPU fitness={gpu_fitness:.6f}, CPU fitness={cpu_fitness:.6f}"
        )
    finally:
        gpu.release()


@skip_no_gpu
def test_gpu_fitness_mae():
    w, h = 30, 30
    target = np.random.rand(h, w, 4).astype(np.float32)
    genes = _fixed_genes()

    gpu = GPURenderer(w, h, target)
    try:
        fitness = gpu.compute_fitness(genes, "mae")
        assert 0 < fitness <= 1.0
    finally:
        gpu.release()


def test_gpu_fallback_to_cpu(monkeypatch):
    from main import create_renderer
    from render.cpu_renderer import CPURenderer
    from config.config_loader import Config

    monkeypatch.setattr("render.gpu_renderer.gpu_available", lambda: False)

    config = Config(
        triangle_count=3, population_size=6, max_generations=5,
        fitness_threshold=2.0, selection_method="Elite",
        crossover_methods=["OnePoint"], crossover_probability=0.7,
        mutation_methods=["Gen"], mutation_rate=0.1,
        survival_strategy="Aditiva", fitness_function="MSE",
        k_offspring=4, save_every=0, use_gpu=True,
    )
    target = np.ones((10, 10, 4), dtype=np.float32)
    renderer = create_renderer(config, target, 10, 10)
    assert isinstance(renderer, CPURenderer)
