import os

import pytest
import numpy as np

from render.gpu_renderer import GPURenderer, _classify_device_info, gpu_available
from genes.triangle_gene import TriangleGene

skip_no_gpu = pytest.mark.skipif(not gpu_available(), reason="GPU no disponible")


class FakeContext:
    def __init__(self, info):
        self.info = info
        self.released = False

    def release(self):
        self.released = True


def _fixed_genes():
    """5 TriangleGene rows for reproducible tests."""
    genes = [
        TriangleGene(x1=0.1, y1=0.1, x2=0.5, y2=0.1, x3=0.3, y3=0.5, r=255, g=0, b=0, a=0.8),
        TriangleGene(x1=0.4, y1=0.3, x2=0.9, y2=0.3, x3=0.6, y3=0.8, r=0, g=255, b=0, a=0.6),
        TriangleGene(x1=0.0, y1=0.5, x2=0.5, y2=0.5, x3=0.25, y3=1.0, r=0, g=0, b=255, a=0.5),
        TriangleGene(x1=0.2, y1=0.0, x2=0.8, y2=0.0, x3=0.5, y3=0.4, r=128, g=128, b=0, a=0.7),
        TriangleGene(x1=0.6, y1=0.6, x2=1.0, y2=0.6, x3=0.8, y3=1.0, r=0, g=128, b=128, a=0.4),
    ]
    return np.array([g.to_row() for g in genes])


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

    monkeypatch.setattr("render.gpu_renderer.gpu_available", lambda preference="auto": False)

    config = Config(
        triangle_count=3, population_size=6, max_generations=5,
        fitness_threshold=2.0, selection_method="Elite",
        crossover_methods=["OnePoint"], crossover_probability=0.7,
        mutation_methods=["Gen"], mutation_rate=0.1,
        survival_strategy="Aditiva", fitness_function="MSE",
        k_offspring=4, save_every=0, use_gpu=True,
        gpu_device="auto",
    )
    target = np.ones((10, 10, 4), dtype=np.float32)
    renderer = create_renderer(config, target, 10, 10)
    assert isinstance(renderer, CPURenderer)


def test_classify_device_info():
    assert _classify_device_info({
        "GL_VENDOR": "NVIDIA Corporation",
        "GL_RENDERER": "NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2",
    }) == ("dedicated", "NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2")

    assert _classify_device_info({
        "GL_VENDOR": "AMD",
        "GL_RENDERER": "AMD Radeon Graphics (radeonsi, renoir, ACO, DRM 3.57)",
    }) == ("integrated", "AMD Radeon Graphics (radeonsi, renoir, ACO, DRM 3.57)")

    assert _classify_device_info({
        "GL_VENDOR": "Intel",
        "GL_RENDERER": "Mesa Intel(R) UHD Graphics 620",
    }) == ("integrated", "Mesa Intel(R) UHD Graphics 620")


def test_gpu_available_dedicated_activa_prime_offload(monkeypatch):
    import moderngl

    seen = {}

    def fake_create_standalone_context():
        seen["prime"] = os.environ.get("__NV_PRIME_RENDER_OFFLOAD")
        seen["vendor"] = os.environ.get("__GLX_VENDOR_LIBRARY_NAME")
        return FakeContext({
            "GL_VENDOR": "NVIDIA Corporation",
            "GL_RENDERER": "NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2",
        })

    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    monkeypatch.setattr(moderngl, "create_standalone_context", fake_create_standalone_context)

    assert gpu_available("dedicated")
    assert seen == {"prime": "1", "vendor": "nvidia"}
    assert os.environ.get("__NV_PRIME_RENDER_OFFLOAD") is None
    assert os.environ.get("__GLX_VENDOR_LIBRARY_NAME") is None


def test_gpu_available_auto_no_activa_prime_offload(monkeypatch):
    import moderngl

    seen = {}

    def fake_create_standalone_context():
        seen["prime"] = os.environ.get("__NV_PRIME_RENDER_OFFLOAD")
        seen["vendor"] = os.environ.get("__GLX_VENDOR_LIBRARY_NAME")
        return FakeContext({
            "GL_VENDOR": "NVIDIA Corporation",
            "GL_RENDERER": "NVIDIA GeForce RTX 3050 Ti Laptop GPU/PCIe/SSE2",
        })

    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    monkeypatch.setattr(moderngl, "create_standalone_context", fake_create_standalone_context)

    assert gpu_available("auto")
    assert seen == {"prime": None, "vendor": None}


def test_gpu_available_dedicated_rechaza_renderer_integrado(monkeypatch):
    import moderngl

    monkeypatch.setattr(
        moderngl,
        "create_standalone_context",
        lambda: FakeContext({
            "GL_VENDOR": "AMD",
            "GL_RENDERER": "AMD Radeon Graphics (radeonsi, renoir, ACO, DRM 3.57)",
        }),
    )

    assert not gpu_available("dedicated")


def test_create_renderer_pasa_gpu_device_al_renderer(monkeypatch):
    from config.config_loader import Config
    from main import create_renderer

    class FakeGPURenderer:
        detect_calls = []
        init_calls = []

        @staticmethod
        def detect_device(preference="auto"):
            FakeGPURenderer.detect_calls.append(preference)
            return "dedicated: Fake GPU"

        def __init__(self, width, height, target_image, device_preference="auto"):
            FakeGPURenderer.init_calls.append((width, height, device_preference, target_image.shape))

    availability_calls = []

    monkeypatch.setattr("render.gpu_renderer.GPURenderer", FakeGPURenderer)
    monkeypatch.setattr(
        "render.gpu_renderer.gpu_available",
        lambda preference="auto": availability_calls.append(preference) or True,
    )

    config = Config(
        triangle_count=3, population_size=6, max_generations=5,
        fitness_threshold=2.0, selection_method="Elite",
        crossover_methods=["OnePoint"], crossover_probability=0.7,
        mutation_methods=["Gen"], mutation_rate=0.1,
        survival_strategy="Aditiva", fitness_function="MSE",
        k_offspring=4, save_every=0, use_gpu=True,
        gpu_device="dedicated",
    )
    target = np.ones((10, 10, 4), dtype=np.float32)

    renderer = create_renderer(config, target, 10, 10)

    assert isinstance(renderer, FakeGPURenderer)
    assert availability_calls == ["dedicated"]
    assert FakeGPURenderer.detect_calls == ["dedicated"]
    assert FakeGPURenderer.init_calls == [(10, 10, "dedicated", (10, 10, 4))]


def test_create_renderer_dedicated_falla_sin_fallback(monkeypatch):
    from config.config_loader import Config
    from main import create_renderer

    class FakeGPURenderer:
        @staticmethod
        def detect_device(preference="auto"):
            return "integrated: AMD Radeon Graphics"

        def __init__(self, *args, **kwargs):
            raise AssertionError("No deberia construir GPURenderer cuando dedicated no esta disponible")

    monkeypatch.setattr("render.gpu_renderer.GPURenderer", FakeGPURenderer)
    monkeypatch.setattr("render.gpu_renderer.gpu_available", lambda preference="auto": False)

    config = Config(
        triangle_count=3, population_size=6, max_generations=5,
        fitness_threshold=2.0, selection_method="Elite",
        crossover_methods=["OnePoint"], crossover_probability=0.7,
        mutation_methods=["Gen"], mutation_rate=0.1,
        survival_strategy="Aditiva", fitness_function="MSE",
        k_offspring=4, save_every=0, use_gpu=True,
        gpu_device="dedicated",
    )
    target = np.ones((10, 10, 4), dtype=np.float32)

    with pytest.raises(RuntimeError, match="GPU dedicada solicitada pero no disponible"):
        create_renderer(config, target, 10, 10)
