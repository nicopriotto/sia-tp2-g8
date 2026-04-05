import logging
import random
import time

import numpy as np

from config.config_loader import Config
from crossover.crossover_operator import CrossoverOperator
from fitness.fitness_function import FitnessFunction
from mutation.mutation_operator import MutationOperator
from render.renderer import Renderer
from selection.selection_strategy import SelectionStrategy
from survival.survival_strategy import SurvivalStrategy
from core.population import Population

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """Loop principal del algoritmo genetico."""

    def __init__(
        self,
        config: Config,
        target_image: np.ndarray,
        renderer: Renderer,
        fitness_fn: FitnessFunction,
        selection: SelectionStrategy,
        crossover_ops: list[CrossoverOperator],
        mutation_ops: list[MutationOperator],
        survival: SurvivalStrategy,
    ):
        self.config = config
        self.target_image = target_image
        self.renderer = renderer
        self.fitness_fn = fitness_fn
        self.selection = selection
        self.crossover_ops = crossover_ops
        self.mutation_ops = mutation_ops
        self.survival = survival

    def run(self) -> Population:
        """Ejecuta el ciclo evolutivo completo y retorna la poblacion final."""
        config = self.config
        population = Population.random(config.population_size, config.triangle_count)
        population.evaluate_all(self.target_image, self.renderer, self.fitness_fn)

        logger.info(
            "Poblacion inicial | Best: %.4f | Avg: %.4f | Std: %.4f",
            population.best.fitness,
            population.average_fitness,
            population.fitness_std,
        )

        stop_reason: str | None = None

        for generation in range(1, config.max_generations + 1):
            generation_start = time.time()
            parents = self.selection.select(population, config.k_offspring)
            crossover_op = random.choice(self.crossover_ops)
            mutation_op = random.choice(self.mutation_ops)

            children = []
            for index in range(0, len(parents) - 1, 2):
                parent1 = parents[index]
                parent2 = parents[index + 1]

                if random.random() < config.crossover_probability:
                    child1, child2 = crossover_op.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = mutation_op.mutate(child1, generation, config.max_generations)
                child2 = mutation_op.mutate(child2, generation, config.max_generations)
                children.extend([child1, child2])

            if len(parents) % 2 != 0:
                last_child = parents[-1].copy()
                last_child = mutation_op.mutate(last_child, generation, config.max_generations)
                children.append(last_child)

            for child in children:
                child.compute_fitness(self.target_image, self.renderer, self.fitness_fn)

            population = self.survival.apply(population, children, self.selection)

            generation_time = time.time() - generation_start
            logger.info(
                "Gen %d | Best: %.4f | Avg: %.4f | Std: %.4f | Time: %.1fs",
                generation,
                population.best.fitness,
                population.average_fitness,
                population.fitness_std,
                generation_time,
            )

            if generation >= config.max_generations:
                stop_reason = f"Maximo de generaciones alcanzado ({config.max_generations})"
                break
            if population.best.fitness >= config.fitness_threshold:
                stop_reason = (
                    "Fitness threshold alcanzado "
                    f"({population.best.fitness:.4f} >= {config.fitness_threshold})"
                )
                break

        if stop_reason is None:
            stop_reason = "Loop completado sin condicion de corte"

        logger.info("=== FIN ===")
        logger.info("Motivo de corte: %s", stop_reason)
        logger.info("Mejor fitness final: %.4f", population.best.fitness)
        logger.info("Fitness promedio final: %.4f", population.average_fitness)
        logger.info("Desviacion estandar final: %.4f", population.fitness_std)

        return population
