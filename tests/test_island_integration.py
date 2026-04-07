import json
import os

import numpy as np
from PIL import Image

from config.config_loader import Config
from main import _build_island, run_from_paths


def _base_config_dict():
    return {
        "triangle_count": 3,
        "population_size": 6,
        "max_generations": 10,
        "fitness_threshold": 2.0,
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.0,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
        "k_offspring": 4,
        "save_every": 0,
    }


def _setup_run(tmp_path, config_dict):
    """Crea imagen y config en tmp_path, retorna paths."""
    os.chdir(tmp_path)
    img = Image.fromarray(np.full((20, 20, 4), 255, dtype=np.uint8))
    img_path = str(tmp_path / "test.png")
    img.save(img_path)

    config_path = str(tmp_path / "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    return img_path, config_path


def test_run_from_paths_normal(tmp_path):
    """run_from_paths sin islas sigue funcionando."""
    img_path, config_path = _setup_run(tmp_path, _base_config_dict())
    result = run_from_paths(img_path, config_path)
    assert result.best_individual.fitness > 0
    assert result.stop_reason == "generaciones_maximas"


def test_run_from_paths_island_mode(tmp_path):
    """run_from_paths con island_enabled=true corre islas."""
    data = _base_config_dict()
    data.update({
        "island_enabled": True,
        "island_count": 2,
        "island_migration_interval": 5,
        "island_migration_count": 1,
        "island_topology": "ring",
    })
    img_path, config_path = _setup_run(tmp_path, data)
    result = run_from_paths(img_path, config_path)
    assert result.best_individual.fitness > 0
    assert result.stop_reason != ""


def test_build_island_creates_independent_instances(tmp_path):
    """Cada isla tiene su propio renderer y context."""
    os.chdir(tmp_path)
    config = Config(**_base_config_dict())
    target = np.ones((20, 20, 4), dtype=np.float32)

    island_0 = _build_island(config, target, 20, 20, 0)
    island_1 = _build_island(config, target, 20, 20, 1)

    assert island_0.renderer is not island_1.renderer
    assert island_0.context is not island_1.context
    assert island_0.output_dir != island_1.output_dir
