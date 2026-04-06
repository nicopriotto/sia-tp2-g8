"""Tests para operadores ponderados (TASK-036)."""
import random

import numpy as np
import pytest

from config.config_loader import Config, parse_weighted_methods
from core.ga_context import GAContext
from core.genetic_algorithm import GeneticAlgorithm
from main import build_operators
from render.cpu_renderer import CPURenderer


# --- Tests de parse_weighted_methods ---


def test_single_method_string_format():
    """Parsear ["Elite"] retorna nombres=["Elite"] y pesos=[1.0]."""
    names, weights = parse_weighted_methods(["Elite"])
    assert names == ["Elite"]
    assert weights == [1.0]


def test_weights_normalized():
    """Parsear dicts con pesos 3.0 y 7.0 normaliza a 0.3 y 0.7."""
    names, weights = parse_weighted_methods([
        {"method": "A", "weight": 3.0},
        {"method": "B", "weight": 7.0},
    ])
    assert names == ["A", "B"]
    assert abs(weights[0] - 0.3) < 1e-9
    assert abs(weights[1] - 0.7) < 1e-9


def test_mixed_format():
    """Formato mixto (string + dict) funciona correctamente."""
    names, weights = parse_weighted_methods([
        {"method": "Boltzmann", "weight": 0.7},
        "Elite",
    ])
    assert names == ["Boltzmann", "Elite"]
    # 0.7 + 1.0 = 1.7, normalizado: 0.7/1.7 y 1.0/1.7
    assert abs(sum(weights) - 1.0) < 1e-9


def test_zero_weight_raises():
    """Peso cero lanza ValueError."""
    with pytest.raises(ValueError, match="Peso debe ser positivo"):
        parse_weighted_methods([{"method": "X", "weight": 0}])


def test_negative_weight_raises():
    """Peso negativo lanza ValueError."""
    with pytest.raises(ValueError, match="Peso debe ser positivo"):
        parse_weighted_methods([{"method": "X", "weight": -1.0}])


def test_invalid_format_raises():
    """Tipo no reconocido lanza ValueError."""
    with pytest.raises(ValueError, match="Formato de operador no reconocido"):
        parse_weighted_methods([42])


# --- Tests de _choose_operator ---


def test_single_method_always_chosen():
    """Con un solo operador, siempre se elige ese operador."""
    operators = ["unico"]
    weights = [1.0]
    for _ in range(100):
        chosen = GeneticAlgorithm._choose_operator(operators, weights)
        assert chosen == "unico"


def test_two_methods_equal_weight_distribution():
    """Con dos operadores de peso igual, cada uno se elige ~50% de las veces."""
    random.seed(12345)
    operators = ["A", "B"]
    weights = [0.5, 0.5]
    counts = {"A": 0, "B": 0}
    n = 2000
    for _ in range(n):
        chosen = GeneticAlgorithm._choose_operator(operators, weights)
        counts[chosen] += 1
    ratio_a = counts["A"] / n
    ratio_b = counts["B"] / n
    assert 0.40 <= ratio_a <= 0.60, f"A fue elegido {ratio_a:.2%}, esperado ~50%"
    assert 0.40 <= ratio_b <= 0.60, f"B fue elegido {ratio_b:.2%}, esperado ~50%"


def test_skewed_weights_distribution():
    """Con pesos 0.99/0.01, el operador raro aparece entre 0.1% y 4%."""
    random.seed(54321)
    operators = ["frecuente", "raro"]
    weights = [0.99, 0.01]
    counts = {"frecuente": 0, "raro": 0}
    n = 2000
    for _ in range(n):
        chosen = GeneticAlgorithm._choose_operator(operators, weights)
        counts[chosen] += 1
    ratio_raro = counts["raro"] / n
    assert 0.001 <= ratio_raro <= 0.04, f"raro fue elegido {ratio_raro:.2%}, esperado ~1%"


# --- Tests de integracion con el GA completo ---


def _make_weighted_config(**overrides) -> Config:
    """Config minima para tests de operadores ponderados."""
    base = {
        "triangle_count": 3,
        "population_size": 6,
        "max_generations": 20,
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


def _make_ga(config: Config, target: np.ndarray | None = None):
    """Helper: construye un GA listo para ejecutar."""
    target = target if target is not None else np.random.rand(10, 10, 4).astype(np.float32)
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
    return ga


def _run_weighted_ga(tmp_path, monkeypatch, config: Config):
    """Helper: ejecuta un GA completo con operadores ponderados."""
    monkeypatch.chdir(tmp_path)
    ga = _make_ga(config)
    return ga.run()


def test_static_weights_mode_preserves_config_weights():
    config = _make_weighted_config(
        selection_methods=["Elite", "Ruleta"],
        selection_weights=[0.8, 0.2],
        mutation_methods=["Gen", "Completa"],
        mutation_weights=[0.7, 0.3],
    )
    ga = _make_ga(config)
    assert ga.selection_weights == [0.8, 0.2]
    assert ga.mutation_weights == [0.7, 0.3]


def test_adaptive_weights_start_uniform_even_if_config_has_custom_weights():
    config = _make_weighted_config(
        selection_methods=["Elite", "Ruleta"],
        selection_weights=[0.9, 0.1],
        mutation_methods=["Gen", "Completa"],
        mutation_weights=[0.8, 0.2],
        adaptive_operator_weights=True,
    )
    ga = _make_ga(config)
    assert ga.selection_weights == [0.5, 0.5]
    assert ga.mutation_weights == [0.5, 0.5]


def test_adaptive_weight_update_rewards_used_operator():
    updated = GeneticAlgorithm._update_weights([0.5, 0.5], chosen_index=0, improved=True, delta=0.05)
    assert updated[0] > 0.5
    assert updated[1] < 0.5
    assert abs(sum(updated) - 1.0) < 1e-9


def test_adaptive_weight_update_penalizes_used_operator():
    updated = GeneticAlgorithm._update_weights([0.5, 0.5], chosen_index=0, improved=False, delta=0.05)
    assert updated[0] < 0.5
    assert updated[1] > 0.5
    assert abs(sum(updated) - 1.0) < 1e-9


def test_adaptive_weight_update_applies_minimum_floor():
    updated = GeneticAlgorithm._update_weights(
        [GeneticAlgorithm._MIN_ADAPTIVE_WEIGHT, 1.0 - GeneticAlgorithm._MIN_ADAPTIVE_WEIGHT],
        chosen_index=0,
        improved=False,
        delta=0.05,
    )
    assert updated[0] >= GeneticAlgorithm._MIN_ADAPTIVE_WEIGHT
    assert abs(sum(updated) - 1.0) < 1e-9


def test_adaptive_weight_update_single_operator_stays_fixed():
    updated = GeneticAlgorithm._update_weights([1.0], chosen_index=0, improved=False, delta=0.05)
    assert updated == [1.0]


def test_weighted_selection_runs(tmp_path, monkeypatch):
    """GA con seleccion ponderada Elite(0.6) + Ruleta(0.4) ejecuta sin excepciones."""
    config = _make_weighted_config(
        selection_methods=["Elite", "Ruleta"],
        selection_weights=[0.6, 0.4],
    )
    result = _run_weighted_ga(tmp_path, monkeypatch, config)
    assert result.best_individual is not None
    assert result.best_individual.fitness > 0


def test_weighted_crossover_runs(tmp_path, monkeypatch):
    """GA con crossover ponderado OnePoint(0.8) + Uniform(0.2) ejecuta sin excepciones."""
    config = _make_weighted_config(
        crossover_methods=["OnePoint", "Uniform"],
        crossover_weights=[0.8, 0.2],
    )
    result = _run_weighted_ga(tmp_path, monkeypatch, config)
    assert result.best_individual is not None


def test_weighted_mutation_runs(tmp_path, monkeypatch):
    """GA con mutacion ponderada Gen(0.7) + Completa(0.3) ejecuta sin excepciones."""
    config = _make_weighted_config(
        mutation_methods=["Gen", "Completa"],
        mutation_weights=[0.7, 0.3],
    )
    result = _run_weighted_ga(tmp_path, monkeypatch, config)
    assert result.best_individual is not None


def test_adaptive_weights_run_and_return_history(tmp_path, monkeypatch):
    config = _make_weighted_config(
        selection_methods=["Elite", "Ruleta"],
        selection_weights=[0.8, 0.2],
        mutation_methods=["Gen", "Completa"],
        mutation_weights=[0.9, 0.1],
        adaptive_operator_weights=True,
    )
    result = _run_weighted_ga(tmp_path, monkeypatch, config)

    assert result.best_individual is not None
    assert len(result.selection_weight_history) == result.final_generation + 1
    assert len(result.mutation_weight_history) == result.final_generation + 1
    assert result.selection_weight_history[0] == [0.5, 0.5]
    assert result.mutation_weight_history[0] == [0.5, 0.5]
    assert any(weights != [0.5, 0.5] for weights in result.selection_weight_history[1:])
    assert any(weights != [0.5, 0.5] for weights in result.mutation_weight_history[1:])
    assert all(abs(sum(weights) - 1.0) < 1e-9 for weights in result.selection_weight_history)
    assert all(abs(sum(weights) - 1.0) < 1e-9 for weights in result.mutation_weight_history)


def test_static_weights_run_returns_empty_histories(tmp_path, monkeypatch):
    config = _make_weighted_config(
        selection_methods=["Elite", "Ruleta"],
        selection_weights=[0.6, 0.4],
        mutation_methods=["Gen", "Completa"],
        mutation_weights=[0.7, 0.3],
        adaptive_operator_weights=False,
    )
    result = _run_weighted_ga(tmp_path, monkeypatch, config)
    assert result.selection_weight_history == []
    assert result.mutation_weight_history == []
