import json
from dataclasses import dataclass, field
from pathlib import Path

VALID_SELECTION_METHODS = [
    "Elite", "Ruleta", "Universal", "Ranking",
    "Boltzmann", "TorneosDeterministicos", "TorneosProbabilisticos",
]

VALID_CROSSOVER_METHODS = ["OnePoint", "TwoPoint", "Uniform", "Annular", "Aritmetico"]

VALID_MUTATION_METHODS = ["Gen", "MultiGen", "Uniforme", "Completa", "NoUniforme", "Gaussiana"]

VALID_SURVIVAL_STRATEGIES = ["Aditiva", "Exclusiva"]

VALID_FITNESS_FUNCTIONS = ["MSE", "MAE", "GMSD"]


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
    use_gpu: bool = False
    gpu_device: str = "auto"
    min_error: float = 0.0  # 0 = desactivado, cualquier valor > 0 actua como criterio de corte
    gene_type: str = "triangle"  # "triangle" | "ellipse"
    arithmetic_alpha: float = 0.5   # Factor de interpolacion para ArithmeticCrossover
    gaussian_sigma: float = 0.1     # Sigma para GaussianMutation
    adaptive_operator_weights: bool = False
    adaptive_operator_delta: float = 0.05
    # Campos para seleccion ponderada de operadores
    selection_methods: list[str] = field(default_factory=list)
    selection_weights: list[float] = field(default_factory=list)
    crossover_weights: list[float] = field(default_factory=list)
    mutation_weights: list[float] = field(default_factory=list)


def parse_weighted_methods(entries: list) -> tuple[list[str], list[float]]:
    """Parsea una lista de operadores con pesos opcionales.

    Acepta strings planos (peso 1.0 por defecto) y dicts {"method": str, "weight": float}.
    Los pesos se normalizan para que sumen 1.0.

    Returns:
        Tupla (nombres, pesos_normalizados).
    """
    names: list[str] = []
    weights: list[float] = []
    for entry in entries:
        if isinstance(entry, str):
            names.append(entry)
            weights.append(1.0)
        elif isinstance(entry, dict):
            name = entry["method"]
            weight = entry.get("weight", 1.0)
            if weight <= 0:
                raise ValueError(f"Peso debe ser positivo, recibido {weight} para {name}")
            names.append(name)
            weights.append(weight)
        else:
            raise ValueError(f"Formato de operador no reconocido: {entry}")

    # Normalizar pesos para que sumen 1.0
    total = sum(weights)
    weights = [w / total for w in weights]
    return names, weights


def load_config(path: str) -> Config:
    """Carga la configuracion desde un archivo JSON y retorna un objeto Config validado."""
    with open(path, "r") as f:
        data = json.load(f)

    # Parsear operadores de seleccion con pesos opcionales.
    # Soporta formato antiguo (selection_method: str) y nuevo (selection_methods: list).
    if "selection_methods" in data:
        sel_names, sel_weights = parse_weighted_methods(data["selection_methods"])
    elif "selection_method" in data:
        sel_value = data["selection_method"]
        if isinstance(sel_value, list):
            sel_names, sel_weights = parse_weighted_methods(sel_value)
        else:
            sel_names = [sel_value]
            sel_weights = [1.0]
    else:
        sel_names = ["Elite"]
        sel_weights = [1.0]

    # Parsear crossover y mutacion con pesos opcionales
    cx_raw = data.get("crossover_methods", ["OnePoint"])
    cx_names, cx_weights = parse_weighted_methods(cx_raw)

    mut_raw = data.get("mutation_methods", ["Gen"])
    mut_names, mut_weights = parse_weighted_methods(mut_raw)

    config = Config(
        triangle_count=data["triangle_count"],
        population_size=data["population_size"],
        max_generations=data["max_generations"],
        fitness_threshold=data["fitness_threshold"],
        selection_method=sel_names[0],
        crossover_methods=cx_names,
        crossover_probability=data["crossover_probability"],
        mutation_methods=mut_names,
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
        use_gpu=data.get("use_gpu", False),
        gpu_device=data.get("gpu_device", "auto"),
        min_error=data.get("min_error", 0.0),
        gene_type=data.get("gene_type", "triangle"),
        arithmetic_alpha=data.get("arithmetic_alpha", 0.5),
        gaussian_sigma=data.get("gaussian_sigma", 0.1),
        adaptive_operator_weights=data.get("adaptive_operator_weights", False),
        adaptive_operator_delta=data.get("adaptive_operator_delta", 0.05),
        selection_methods=sel_names,
        selection_weights=sel_weights,
        crossover_weights=cx_weights,
        mutation_weights=mut_weights,
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

    for method in config.selection_methods:
        if method not in VALID_SELECTION_METHODS:
            raise ValueError(
                f"selection_method '{method}' no es valido. "
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

    if not (0.0 <= config.generational_gap <= 1.0):
        raise ValueError(
            f"generational_gap debe estar en [0, 1], recibido: {config.generational_gap}"
        )

    if config.adaptive_operator_delta <= 0:
        raise ValueError(
            f"adaptive_operator_delta debe ser > 0, recibido: {config.adaptive_operator_delta}"
        )

    if config.gpu_device not in ("auto", "dedicated", "integrated"):
        raise ValueError(
            f"gpu_device '{config.gpu_device}' no es valido. Opciones: auto, dedicated, integrated"
        )

    valid_gene_types = ("triangle", "ellipse")
    if config.gene_type not in valid_gene_types:
        raise ValueError(
            f"gene_type '{config.gene_type}' no es valido. Opciones: {valid_gene_types}"
        )
