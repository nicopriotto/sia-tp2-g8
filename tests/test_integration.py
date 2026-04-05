import json
import random

import numpy as np
from PIL import Image

from config.config_loader import Config
from core.genetic_algorithm import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from main import run_from_paths
from mutation.gen_mutation import GenMutation
from render.cpu_renderer import CPURenderer
from selection.elite import EliteSelection
from survival.additive import AdditiveSurvival


def _make_config(**overrides) -> Config:
    base = {
        "triangle_count": 5,
        "population_size": 10,
        "max_generations": 20,
        "fitness_threshold": 2.0,
        "selection_method": "elite",
        "crossover_methods": ["one_point"],
        "crossover_probability": 0.0,
        "mutation_methods": ["gen"],
        "mutation_rate": 1.0,
        "survival_strategy": "generational",
        "fitness_function": "mse",
        "k_offspring": 6,
        "save_every": 0,
    }
    base.update(overrides)
    return Config(**base)


def _bad_gene():
    from genes.triangle_gene import TriangleGene

    return TriangleGene(
        x1=0.0,
        y1=0.0,
        x2=1.0,
        y2=0.0,
        x3=0.0,
        y3=1.0,
        r=0,
        g=0,
        b=0,
        a=1.0,
    )


def _good_gene():
    from genes.triangle_gene import TriangleGene

    return TriangleGene(
        x1=0.0,
        y1=0.0,
        x2=1.0,
        y2=0.0,
        x3=0.0,
        y3=1.0,
        r=255,
        g=255,
        b=255,
        a=0.0,
    )


def _initial_population(size: int, triangle_count: int) -> Population:
    individuals = []
    for _ in range(size):
        genes = [_bad_gene()] + [_good_gene() for _ in range(triangle_count - 1)]
        individuals.append(Individual(genes=genes))
    return Population(individuals=individuals)


def test_fitness_increases_over_generations(monkeypatch):
    from genes.triangle_gene import TriangleGene

    random.seed(123)
    np.random.seed(123)

    target = np.ones((50, 50, 4), dtype=np.float32)
    config = _make_config()
    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    selection = EliteSelection()
    crossover_ops = [OnePointCrossover()]
    mutation_ops = [GenMutation(config.mutation_rate)]
    survival = AdditiveSurvival()

    initial_population = _initial_population(config.population_size, config.triangle_count)
    initial_population.evaluate_all(target, renderer, fitness_fn)
    initial_best = initial_population.best.fitness

    def fake_population_random(cls, size, n_triangles):
        individuals = [individual.copy() for individual in initial_population.individuals]
        return Population(individuals=individuals)

    monkeypatch.setattr(Population, "random", classmethod(fake_population_random))
    monkeypatch.setattr(TriangleGene, "mutate_replace", lambda self: _good_gene().copy())
    monkeypatch.setattr("mutation.gen_mutation.random.randint", lambda low, high: 0)

    ga = GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=fitness_fn,
        selection=selection,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
    )

    result = ga.run()

    assert result.best_individual.fitness > initial_best


def test_output_files_created(tmp_path, monkeypatch):
    image_path = tmp_path / "target.png"
    config_path = tmp_path / "config.json"

    target = np.ones((50, 50, 4), dtype=np.uint8) * 255
    Image.fromarray(target).save(image_path)

    config_data = {
        "triangle_count": 3,
        "population_size": 6,
        "max_generations": 3,
        "fitness_threshold": 2.0,
        "selection_method": "elite",
        "crossover_methods": ["one_point"],
        "crossover_probability": 0.7,
        "mutation_methods": ["gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "generational",
        "fitness_function": "mse",
        "k_offspring": 4,
        "save_every": 0,
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    random.seed(321)
    np.random.seed(321)

    run_from_paths(str(image_path), str(config_path))

    assert (tmp_path / "output" / "final.png").exists()
    assert (tmp_path / "output" / "triangles.json").exists()


def test_triangles_json_structure(tmp_path, monkeypatch):
    image_path = tmp_path / "target.png"
    config_path = tmp_path / "config.json"

    target = np.ones((50, 50, 4), dtype=np.uint8) * 255
    Image.fromarray(target).save(image_path)

    config_data = {
        "triangle_count": 4,
        "population_size": 6,
        "max_generations": 2,
        "fitness_threshold": 2.0,
        "selection_method": "elite",
        "crossover_methods": ["one_point"],
        "crossover_probability": 0.7,
        "mutation_methods": ["gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "generational",
        "fitness_function": "mse",
        "k_offspring": 4,
        "save_every": 0,
    }
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    random.seed(456)
    np.random.seed(456)

    run_from_paths(str(image_path), str(config_path))

    data = json.loads((tmp_path / "output" / "triangles.json").read_text(encoding="utf-8"))

    assert "genes" in data
    assert "fitness" in data
    assert len(data["genes"]) == config_data["triangle_count"]
    expected_keys = {"x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a"}
    assert set(data["genes"][0].keys()) == expected_keys
