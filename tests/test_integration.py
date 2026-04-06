import json
import random

import numpy as np
import pytest
from PIL import Image

from config.config_loader import Config
from core.ga_context import GAContext
from core.genetic_algorithm import GeneticAlgorithm
from core.individual import Individual
from core.population import Population
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from main import run_from_paths, build_operators, SELECTIONS, CROSSOVERS, MUTATIONS, SURVIVALS, FITNESS
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
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.0,
        "mutation_methods": ["Gen"],
        "mutation_rate": 1.0,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
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

    def fake_population_random(cls, size, n_triangles, gene_type="triangle"):
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
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.7,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
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
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.7,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.1,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
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
    expected_keys = {"x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a", "active"}
    assert set(data["genes"][0].keys()) == expected_keys


# --- Tests de integracion del factory ---

def _make_factory_config(**overrides) -> Config:
    """Config minima para tests de factory con 5 generaciones rapidas."""
    base = {
        "triangle_count": 3,
        "population_size": 6,
        "max_generations": 5,
        "fitness_threshold": 2.0,
        "selection_method": "Elite",
        "crossover_methods": ["OnePoint"],
        "crossover_probability": 0.7,
        "mutation_methods": ["Gen"],
        "mutation_rate": 0.3,
        "survival_strategy": "Aditiva",
        "fitness_function": "MSE",
        "k_offspring": 4,
        "save_every": 0,
    }
    base.update(overrides)
    return Config(**base)


def _run_ga_with_config(tmp_path, monkeypatch, config: Config):
    """Helper: ejecuta el GA con una config dada y retorna el resultado."""
    monkeypatch.chdir(tmp_path)
    target = np.random.rand(10, 10, 4).astype(np.float32)
    renderer = CPURenderer()

    selection_ops, crossover_ops, mutation_ops, survival, fitness = build_operators(config)
    context = GAContext(generation=0, max_generations=config.max_generations)

    ga = GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=fitness,
        selection_ops=selection_ops,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
        context=context,
    )
    return ga.run()


@pytest.mark.parametrize("sel_name", list(SELECTIONS.keys()))
def test_all_selection_methods_run(sel_name, tmp_path, monkeypatch):
    overrides = {"selection_method": sel_name}
    if sel_name == "Boltzmann":
        overrides.update(boltzmann_t0=100.0, boltzmann_tc=1.0, boltzmann_k=0.05)
    elif sel_name == "TorneosDeterministicos":
        overrides["tournament_m"] = 3
    elif sel_name == "TorneosProbabilisticos":
        overrides["tournament_threshold"] = 0.75

    config = _make_factory_config(**overrides)
    result = _run_ga_with_config(tmp_path, monkeypatch, config)
    assert result.best_individual is not None
    assert result.best_individual.fitness > 0


@pytest.mark.parametrize("cx_name", list(CROSSOVERS.keys()))
def test_all_crossover_methods_run(cx_name, tmp_path, monkeypatch):
    config = _make_factory_config(crossover_methods=[cx_name])
    result = _run_ga_with_config(tmp_path, monkeypatch, config)
    assert result.best_individual is not None


@pytest.mark.parametrize("mut_name", list(MUTATIONS.keys()))
def test_all_mutation_methods_run(mut_name, tmp_path, monkeypatch):
    config = _make_factory_config(mutation_methods=[mut_name])
    result = _run_ga_with_config(tmp_path, monkeypatch, config)
    assert result.best_individual is not None


@pytest.mark.parametrize("surv_name", list(SURVIVALS.keys()))
def test_both_survival_strategies_run(surv_name, tmp_path, monkeypatch):
    config = _make_factory_config(survival_strategy=surv_name)
    result = _run_ga_with_config(tmp_path, monkeypatch, config)
    assert result.best_individual is not None


@pytest.mark.parametrize("fit_name", list(FITNESS.keys()))
def test_both_fitness_functions_run(fit_name, tmp_path, monkeypatch):
    config = _make_factory_config(fitness_function=fit_name)
    result = _run_ga_with_config(tmp_path, monkeypatch, config)
    assert result.best_individual is not None
    assert 0 < result.best_individual.fitness <= 1.0
