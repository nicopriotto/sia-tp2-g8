from dataclasses import dataclass, field

from core.individual import Individual


@dataclass
class GAResult:
    best_individual: Individual
    final_generation: int
    stop_reason: str
    elapsed_seconds: float
    best_fitness_history: list[float] = field(default_factory=list)
    selection_weight_history: list[list[float]] = field(default_factory=list)
    mutation_weight_history: list[list[float]] = field(default_factory=list)
