from dataclasses import dataclass


@dataclass
class GAContext:
    generation: int = 0
    max_generations: int = 0
