import logging
import random
import time

import numpy as np

from config.config_loader import Config
from core.ga_result import GAResult
from core.metrics_collector import MetricsCollector
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

    def run(self) -> GAResult:
        """Ejecuta el ciclo evolutivo completo y retorna un GAResult."""
        config = self.config
        height, width = self.target_image.shape[0], self.target_image.shape[1]

        collector = MetricsCollector(
            output_dir="output",
            save_every=config.save_every,
            renderer=self.renderer,
            width=width,
            height=height,
        )
        collector.init_csv()

        start_time = time.time()

        population = Population.random(config.population_size, config.triangle_count)
        population.evaluate_all(self.target_image, self.renderer, self.fitness_fn)

        collector.log_generation(0, population, time.time() - start_time)
        collector.save_snapshot(0, population.best)

        logger.info(
            "Poblacion inicial | Best: %.4f | Avg: %.4f | Std: %.4f",
            population.best.fitness,
            population.average_fitness,
            population.fitness_std,
        )

        k_offspring = max(2, int(config.generational_gap * config.population_size))
        if k_offspring % 2 != 0:
            k_offspring += 1

        stop_reason: str | None = None
        last_generation = 0
        best_fitness_history: list[float] = [population.best.fitness]
        low_diversity_counter = 0

        for generation in range(1, config.max_generations + 1):
            generation_start = time.time()

            # Criterio: tiempo maximo
            if config.max_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= config.max_seconds:
                    stop_reason = "tiempo_maximo"
                    last_generation = generation - 1
                    break

            parents = self.selection.select(population, k_offspring, generation=generation)
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

            best_fitness_history.append(population.best.fitness)

            elapsed = time.time() - start_time
            collector.log_generation(generation, population, elapsed)
            collector.save_snapshot(generation, population.best)

            generation_time = time.time() - generation_start
            logger.info(
                "Gen %d | Best: %.4f | Avg: %.4f | Std: %.4f | Time: %.1fs",
                generation,
                population.best.fitness,
                population.average_fitness,
                population.fitness_std,
                generation_time,
            )

            last_generation = generation

            # Criterio: fitness threshold
            if population.best.fitness >= config.fitness_threshold:
                stop_reason = "fitness_alcanzado"
                break

            # Criterio: contenido
            if config.content_generations > 0 and len(best_fitness_history) >= config.content_generations:
                recent = best_fitness_history[-config.content_generations:]
                if max(recent) - min(recent) < config.content_threshold:
                    stop_reason = "contenido"
                    break

            # Criterio: estructura
            if config.structure_generations > 0:
                if population.fitness_std < config.structure_threshold:
                    low_diversity_counter += 1
                else:
                    low_diversity_counter = 0
                if low_diversity_counter >= config.structure_generations:
                    stop_reason = "estructura"
                    break

            # Criterio: generaciones maximas
            if generation >= config.max_generations:
                stop_reason = "generaciones_maximas"
                break

        if stop_reason is None:
            stop_reason = "generaciones_maximas"

        elapsed_seconds = time.time() - start_time

        collector.save_final_result(population.best, config, last_generation)
        collector.save_final_image(population.best)

        logger.info("=== FIN ===")
        logger.info("Motivo de corte: %s", stop_reason)
        logger.info("Mejor fitness final: %.4f", population.best.fitness)
        logger.info("Fitness promedio final: %.4f", population.average_fitness)
        logger.info("Desviacion estandar final: %.4f", population.fitness_std)

        return GAResult(
            best_individual=population.best,
            final_generation=last_generation,
            stop_reason=stop_reason,
            elapsed_seconds=elapsed_seconds,
            best_fitness_history=best_fitness_history,
        )
