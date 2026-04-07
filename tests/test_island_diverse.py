import json
import os

import numpy as np
import pytest

from config.config_loader import Config, apply_island_overrides
from main import _build_island


def _base_config():
    return Config(
        triangle_count=5,
        population_size=10,
        max_generations=50,
        fitness_threshold=0.99,
        selection_method="Boltzmann",
        crossover_methods=["OnePoint"],
        crossover_probability=0.7,
        mutation_methods=["Gen"],
        mutation_rate=0.05,
        survival_strategy="Aditiva",
        fitness_function="MSE",
        k_offspring=8,
        save_every=10,
        selection_methods=["Boltzmann"],
        selection_weights=[1.0],
        crossover_weights=[1.0],
        mutation_weights=[1.0],
    )


def test_override_mutation_rate():
    """Override de mutation_rate se aplica, resto queda igual."""
    base = _base_config()
    new = apply_island_overrides(base, {"mutation_rate": 0.15})
    assert new.mutation_rate == 0.15
    assert new.population_size == base.population_size
    assert new.selection_method == base.selection_method
    assert new.fitness_function == base.fitness_function


def test_override_selection_method():
    """Override de selection_method actualiza selection_methods tambien."""
    base = _base_config()
    new = apply_island_overrides(base, {"selection_method": "Elite"})
    assert new.selection_method == "Elite"
    assert new.selection_methods == ["Elite"]
    assert new.selection_weights == [1.0]


def test_reject_global_field():
    """Override de campo global lanza ValueError."""
    base = _base_config()
    with pytest.raises(ValueError, match="campo global"):
        apply_island_overrides(base, {"island_count": 10})


def test_reject_nonexistent_field():
    """Override de campo inexistente lanza ValueError."""
    base = _base_config()
    with pytest.raises(ValueError, match="no existe en Config"):
        apply_island_overrides(base, {"campo_falso": 42})


def test_name_field_ignored():
    """El campo 'name' no lanza error, se filtra."""
    base = _base_config()
    new = apply_island_overrides(base, {"name": "exploradora", "mutation_rate": 0.1})
    assert new.mutation_rate == 0.1


def test_override_mutation_methods_with_weights():
    """Override de mutation_methods re-parsea pesos correctamente."""
    base = _base_config()
    new = apply_island_overrides(base, {"mutation_methods": ["Gen", "Completa"]})
    assert new.mutation_methods == ["Gen", "Completa"]
    assert len(new.mutation_weights) == 2
    assert abs(sum(new.mutation_weights) - 1.0) < 1e-9


def test_build_island_with_overrides(tmp_path):
    """_build_island con overrides crea GA con config diferente al base."""
    os.chdir(tmp_path)
    base = _base_config()
    target = np.ones((20, 20, 4), dtype=np.float32)

    island_0 = _build_island(base, target, 20, 20, 0, overrides={"mutation_rate": 0.15})
    island_1 = _build_island(base, target, 20, 20, 1, overrides=None)

    assert island_0.config.mutation_rate == 0.15
    assert island_1.config.mutation_rate == 0.05


def test_build_island_with_name_override(tmp_path):
    """_build_island con name usa el nombre en output_dir."""
    os.chdir(tmp_path)
    base = _base_config()
    target = np.ones((20, 20, 4), dtype=np.float32)

    island = _build_island(base, target, 20, 20, 0, overrides={"name": "exploradora", "mutation_rate": 0.1})
    assert "exploradora" in island.output_dir


def test_diverse_islands_end_to_end(tmp_path):
    """run_from_paths con island_configs corre islas diversas."""
    from main import run_from_paths
    from PIL import Image

    os.chdir(tmp_path)
    img = Image.fromarray(np.full((20, 20, 4), 255, dtype=np.uint8))
    img_path = str(tmp_path / "test.png")
    img.save(img_path)

    data = {
        "triangle_count": 3,
        "population_size": 6,
        "max_generations": 10,
        "fitness_threshold": 2.0,
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.0,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.05,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
        "k_offspring": 4,
        "save_every": 0,
        "island_enabled": True,
        "island_count": 2,
        "island_migration_interval": 5,
        "island_migration_count": 1,
        "island_topology": "ring",
        "island_configs": [
            {"name": "alta-mutacion", "mutation_rate": 0.2},
            {"name": "baja-mutacion", "mutation_rate": 0.01},
        ],
    }
    config_path = str(tmp_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(data, f)

    result = run_from_paths(img_path, config_path)
    assert result.best_individual.fitness > 0
