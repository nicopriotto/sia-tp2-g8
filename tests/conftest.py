import pytest
import numpy as np
from config.config_loader import Config


@pytest.fixture
def sample_config() -> Config:
    """Configuracion minima valida para tests."""
    return Config(
        triangle_count=5,
        population_size=10,
        max_generations=50,
        fitness_threshold=0.99,
        selection_method="elite",
        crossover_methods=["one_point"],
        crossover_probability=0.7,
        mutation_methods=["gen"],
        mutation_rate=0.1,
        survival_strategy="generational",
        fitness_function="mse",
        k_offspring=8,
        save_every=10,
    )


@pytest.fixture
def sample_image() -> np.ndarray:
    """Imagen RGBA 100x100 blanca, float32, rango [0,1]."""
    return np.ones((100, 100, 4), dtype=np.float32)


@pytest.fixture
def sample_triangle():
    """Importacion lazy para evitar dependencia circular. Retorna un TriangleGene aleatorio."""
    from genes.triangle_gene import TriangleGene
    return TriangleGene.random()
