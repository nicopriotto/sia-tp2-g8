from unittest.mock import MagicMock
import numpy as np
import pytest

from core.individual import Individual
from fitness.mse import MSEFitness
from fitness.mae import MAEFitness
from render.cpu_renderer import CPURenderer


@pytest.fixture
def target_image():
    return np.ones((64, 64, 4), dtype=np.float32)


@pytest.fixture
def individual():
    return Individual.random(3, gene_type="triangle")


def test_cpu_path_usa_fitness_fn(individual, target_image):
    """Con CPURenderer, el flujo render + fitness_fn.compute funciona sin errores."""
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    individual.compute_fitness(target_image, renderer, fitness_fn)
    assert individual.fitness > 0
    assert individual.fitness_valid is True


def test_cpu_renderer_compute_fitness_retorna_none():
    """CPURenderer.compute_fitness() retorna None (no soporta fitness en GPU)."""
    renderer = CPURenderer()
    genes = np.zeros((3, 11), dtype=np.float32)
    result = renderer.compute_fitness(genes, "mse")
    assert result is None


def test_fitness_function_tiene_name():
    """MSEFitness y MAEFitness tienen atributo name correcto."""
    assert MSEFitness().name == "mse"
    assert MAEFitness().name == "mae"


def test_gpu_path_usa_renderer_compute_fitness(individual, target_image):
    """Cuando renderer.compute_fitness() retorna un valor, se usa directamente."""
    renderer = MagicMock()
    renderer.compute_fitness.return_value = 0.75
    fitness_fn = MSEFitness()

    individual.compute_fitness(target_image, renderer, fitness_fn)

    assert individual.fitness == 0.75
    assert individual.fitness_valid is True
    renderer.compute_fitness.assert_called_once_with(
        individual.genes, fitness_type="mse", gene_type="triangle"
    )


def test_gpu_path_no_llama_fitness_fn(individual, target_image):
    """Cuando se usa el path GPU, fitness_fn.compute() nunca se llama."""
    renderer = MagicMock()
    renderer.compute_fitness.return_value = 0.80

    fitness_fn = MagicMock()
    fitness_fn.name = "mse"
    fitness_fn.compute.side_effect = AssertionError("No deberia llamarse")

    individual.compute_fitness(target_image, renderer, fitness_fn)

    assert individual.fitness == 0.80
    fitness_fn.compute.assert_not_called()
    renderer.render.assert_not_called()
