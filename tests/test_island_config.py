import json
import os

import pytest

from config.config_loader import Config, load_config


def _base_config_dict():
    """Dict minimo valido para crear un Config via load_config."""
    return {
        "triangle_count": 5,
        "population_size": 10,
        "max_generations": 50,
        "fitness_threshold": 0.99,
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.7,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
        "k_offspring": 8,
        "save_every": 10,
    }


def _write_config(tmp_path, data):
    path = os.path.join(tmp_path, "config.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_defaults_island_disabled(tmp_path):
    """Config sin campos de isla tiene island_enabled=False y defaults correctos."""
    path = _write_config(tmp_path, _base_config_dict())
    config = load_config(path)
    assert config.island_enabled is False
    assert config.island_count == 5
    assert config.island_migration_interval == 50
    assert config.island_migration_count == 2
    assert config.island_topology == "ring"
    assert config.island_configs == []


def test_parse_island_fields(tmp_path):
    """Campos de isla se parsean correctamente desde JSON."""
    data = _base_config_dict()
    data.update({
        "island_enabled": True,
        "island_count": 3,
        "island_migration_interval": 25,
        "island_migration_count": 1,
        "island_topology": "fully_connected",
    })
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.island_enabled is True
    assert config.island_count == 3
    assert config.island_migration_interval == 25
    assert config.island_migration_count == 1
    assert config.island_topology == "fully_connected"


def test_validate_island_count_too_low(tmp_path):
    """island_count < 2 con island_enabled=True debe fallar."""
    data = _base_config_dict()
    data.update({"island_enabled": True, "island_count": 1})
    path = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="island_count debe ser >= 2"):
        load_config(path)


def test_validate_migration_count_too_high(tmp_path):
    """island_migration_count >= population_size debe fallar."""
    data = _base_config_dict()
    data.update({
        "island_enabled": True,
        "island_count": 3,
        "island_migration_count": 10,  # == population_size
    })
    path = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="island_migration_count"):
        load_config(path)


def test_validate_invalid_topology(tmp_path):
    """Topologia invalida debe fallar."""
    data = _base_config_dict()
    data.update({
        "island_enabled": True,
        "island_count": 3,
        "island_topology": "estrella",
    })
    path = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="island_topology"):
        load_config(path)


def test_validate_island_configs_wrong_length(tmp_path):
    """island_configs con largo != island_count debe fallar."""
    data = _base_config_dict()
    data.update({
        "island_enabled": True,
        "island_count": 3,
        "island_configs": [
            {"mutation_rate": 0.1},
            {"mutation_rate": 0.2},
        ],
    })
    path = _write_config(tmp_path, data)
    with pytest.raises(ValueError, match="island_configs tiene 2 entradas"):
        load_config(path)
