from unittest.mock import MagicMock
import numpy as np
import pytest

from core.individual import Individual
from fitness.mse import MSEFitness
from fitness.mae import MAEFitness
from fitness.ssim import SSIMFitness
from render.cpu_renderer import CPURenderer


@pytest.fixture
def target_image():
    return np.ones((64, 64, 4), dtype=np.float32)


@pytest.fixture
def individual():
    return Individual.random(3, gene_type="triangle")


@pytest.fixture
def individual_ellipse():
    return Individual.random(3, gene_type="ellipse")


def test_cpu_renderer_fallback(individual, target_image):
    """Con CPURenderer, individual.compute_fitness sigue funcionando (path CPU)."""
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    individual.compute_fitness(target_image, renderer, fitness_fn)
    assert individual.fitness > 0
    assert individual.fitness_valid is True


def test_fitness_name_attributes():
    """Todas las fitness functions tienen name correcto."""
    assert MSEFitness().name == "mse"
    assert MAEFitness().name == "mae"
    assert SSIMFitness().name == "ssim"


def test_gpu_path_routes_ssim(individual, target_image):
    """Cuando renderer soporta compute_fitness con ssim, se usa el path GPU."""
    renderer = MagicMock()
    renderer.compute_fitness.return_value = 0.85
    fitness_fn = SSIMFitness()

    individual.compute_fitness(target_image, renderer, fitness_fn)

    assert individual.fitness == 0.85
    renderer.compute_fitness.assert_called_once_with(
        individual.genes, fitness_type="ssim", gene_type="triangle"
    )


def test_gpu_path_routes_mse(individual, target_image):
    """Cuando renderer soporta compute_fitness con mse, se usa el path GPU."""
    renderer = MagicMock()
    renderer.compute_fitness.return_value = 0.92
    fitness_fn = MSEFitness()

    individual.compute_fitness(target_image, renderer, fitness_fn)

    assert individual.fitness == 0.92
    renderer.compute_fitness.assert_called_once_with(
        individual.genes, fitness_type="mse", gene_type="triangle"
    )


def test_gpu_path_routes_mse_ellipse(individual_ellipse, target_image):
    """Cuando el individuo es ellipse, propaga gene_type al path GPU."""
    renderer = MagicMock()
    renderer.compute_fitness.return_value = 0.77
    fitness_fn = MSEFitness()

    individual_ellipse.compute_fitness(target_image, renderer, fitness_fn)

    assert individual_ellipse.fitness == 0.77
    renderer.compute_fitness.assert_called_once_with(
        individual_ellipse.genes, fitness_type="mse", gene_type="ellipse"
    )
