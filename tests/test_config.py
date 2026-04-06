import json
import pytest
import tempfile
from config.config_loader import load_config


def _write_config(data: dict) -> str:
    """Helper: escribe un dict como JSON temporal y retorna el path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


VALID_DATA = {
    "triangle_count": 10,
    "population_size": 20,
    "max_generations": 100,
    "fitness_threshold": 0.95,
    "selection_method": "Elite",
    "crossover_methods": ["OnePoint"],
    "crossover_probability": 0.7,
    "mutation_methods": ["Gen"],
    "mutation_rate": 0.05,
    "survival_strategy": "Aditiva",
    "fitness_function": "MSE",
    "k_offspring": 16,
    "save_every": 10,
}


def test_load_valid_config():
    path = _write_config(VALID_DATA)
    config = load_config(path)
    assert config.triangle_count == 10
    assert config.population_size == 20
    assert config.selection_method == "Elite"


def test_invalid_selection_method():
    data = {**VALID_DATA, "selection_method": "inexistente"}
    path = _write_config(data)
    with pytest.raises(ValueError, match="selection_method"):
        load_config(path)


def test_invalid_fitness_function():
    data = {**VALID_DATA, "fitness_function": "cosine"}
    path = _write_config(data)
    with pytest.raises(ValueError, match="fitness_function"):
        load_config(path)


def test_defaults_applied():
    path = _write_config(VALID_DATA)
    config = load_config(path)
    assert config.boltzmann_t0 == 100.0
    assert config.boltzmann_tc == 1.0
    assert config.boltzmann_k == 0.01
    assert config.tournament_m == 5
    assert config.tournament_threshold == 0.75
    assert config.generational_gap == 1.0
    assert config.max_seconds == 0.0
