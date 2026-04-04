import json
from dataclasses import dataclass
from pathlib import Path

VALID_SELECTION_METHODS = [
    "elite", "roulette", "universal", "ranking",
    "boltzmann", "tournament_det", "tournament_prob"
]

VALID_CROSSOVER_METHODS = ["one_point", "two_point", "uniform", "anular"]

VALID_MUTATION_METHODS = ["gen", "multigen"]

VALID_SURVIVAL_STRATEGIES = ["generational", "youth_biased"]

VALID_FITNESS_FUNCTIONS = ["mse", "mae"]


@dataclass
class Config:
    triangle_count: int
    population_size: int
    max_generations: int
    fitness_threshold: float
    selection_method: str
    crossover_methods: list[str]
    crossover_probability: float
    mutation_methods: list[str]
    mutation_rate: float
    survival_strategy: str
    fitness_function: str
    k_offspring: int
    save_every: int
    boltzmann_t0: float = 100.0
    boltzmann_tc: float = 1.0
    boltzmann_k: float = 0.01
    tournament_m: int = 5
    tournament_threshold: float = 0.75
    non_uniform_b: float = 1.0
    generational_gap: float = 1.0
    max_seconds: float = 0.0
    content_threshold: float = 0.0
    content_generations: int = 0
    structure_threshold: float = 0.0
    structure_generations: int = 0


def load_config(path: str) -> Config:
    """Carga la configuracion desde un archivo JSON y retorna un objeto Config validado."""
    with open(path, "r") as f:
        data = json.load(f)

    config = Config(
        triangle_count=data["triangle_count"],
        population_size=data["population_size"],
        max_generations=data["max_generations"],
        fitness_threshold=data["fitness_threshold"],
        selection_method=data["selection_method"],
        crossover_methods=data["crossover_methods"],
        crossover_probability=data["crossover_probability"],
        mutation_methods=data["mutation_methods"],
        mutation_rate=data["mutation_rate"],
        survival_strategy=data["survival_strategy"],
        fitness_function=data["fitness_function"],
        k_offspring=data["k_offspring"],
        save_every=data["save_every"],
        boltzmann_t0=data.get("boltzmann_t0", 100.0),
        boltzmann_tc=data.get("boltzmann_tc", 1.0),
        boltzmann_k=data.get("boltzmann_k", 0.01),
        tournament_m=data.get("tournament_m", 5),
        tournament_threshold=data.get("tournament_threshold", 0.75),
        non_uniform_b=data.get("non_uniform_b", 1.0),
        generational_gap=data.get("generational_gap", 1.0),
        max_seconds=data.get("max_seconds", 0.0),
        content_threshold=data.get("content_threshold", 0.0),
        content_generations=data.get("content_generations", 0),
        structure_threshold=data.get("structure_threshold", 0.0),
        structure_generations=data.get("structure_generations", 0),
    )

    _validate_config(config)
    return config


def _validate_config(config: Config) -> None:
    """Valida que los valores de string sean opciones conocidas."""
    if config.selection_method not in VALID_SELECTION_METHODS:
        raise ValueError(
            f"selection_method '{config.selection_method}' no es valido. "
            f"Opciones: {VALID_SELECTION_METHODS}"
        )

    for method in config.crossover_methods:
        if method not in VALID_CROSSOVER_METHODS:
            raise ValueError(
                f"crossover_method '{method}' no es valido. "
                f"Opciones: {VALID_CROSSOVER_METHODS}"
            )

    for method in config.mutation_methods:
        if method not in VALID_MUTATION_METHODS:
            raise ValueError(
                f"mutation_method '{method}' no es valido. "
                f"Opciones: {VALID_MUTATION_METHODS}"
            )

    if config.survival_strategy not in VALID_SURVIVAL_STRATEGIES:
        raise ValueError(
            f"survival_strategy '{config.survival_strategy}' no es valido. "
            f"Opciones: {VALID_SURVIVAL_STRATEGIES}"
        )

    if config.fitness_function not in VALID_FITNESS_FUNCTIONS:
        raise ValueError(
            f"fitness_function '{config.fitness_function}' no es valido. "
            f"Opciones: {VALID_FITNESS_FUNCTIONS}"
        )
