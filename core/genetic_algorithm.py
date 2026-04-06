import logging
import random
import time

import numpy as np

from config.config_loader import Config
from core.ga_context import GAContext
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
        selection: SelectionStrategy | None = None,
        crossover_ops: list[CrossoverOperator] | None = None,
        mutation_ops: list[MutationOperator] | None = None,
        survival: SurvivalStrategy | None = None,
        context: GAContext | None = None,
        output_dir: str = "output",
        selection_ops: list[SelectionStrategy] | None = None,
    ):
        self.config = config
        self.target_image = target_image
        self.renderer = renderer
        self.fitness_fn = fitness_fn

        # Soportar seleccion multiple ponderada.
        # Si se pasa selection_ops (lista), usarla; sino usar selection (singular, backward compat).
        if selection_ops is not None:
            self.selection_ops = selection_ops
        elif selection is not None:
            self.selection_ops = [selection]
        else:
            raise ValueError("Se requiere 'selection' o 'selection_ops'")

        self.crossover_ops = crossover_ops or []
        self.mutation_ops = mutation_ops or []
        self.survival = survival
        self.context = context
        self.output_dir = output_dir

        # Pesos para seleccion ponderada de operadores
        self.selection_weights = config.selection_weights if config.selection_weights else [1.0] * len(self.selection_ops)
        self.crossover_weights = config.crossover_weights if config.crossover_weights else [1.0] * len(self.crossover_ops)
        self.mutation_weights = config.mutation_weights if config.mutation_weights else [1.0] * len(self.mutation_ops)

    @staticmethod
    def _choose_operator(operators: list, weights: list):
        """Elige un operador segun pesos ponderados."""
        return random.choices(operators, weights=weights, k=1)[0]

    def run(self) -> GAResult:
        """Ejecuta el ciclo evolutivo completo y retorna un GAResult."""
        config = self.config
        height, width = self.target_image.shape[0], self.target_image.shape[1]

        collector = MetricsCollector(
            output_dir=self.output_dir,
            save_every=config.save_every,
            renderer=self.renderer,
            width=width,
            height=height,
        )
        collector.init_csv()

        start_time = time.time()

        gene_type = getattr(config, 'gene_type', 'triangle')
        population = Population.random(config.population_size, config.triangle_count, gene_type)
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

            if self.context is not None:
                self.context.generation = generation

            # Criterio: tiempo maximo
            if config.max_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= config.max_seconds:
                    stop_reason = "tiempo_maximo"
                    last_generation = generation - 1
                    break

            selector = self._choose_operator(self.selection_ops, self.selection_weights)
            parents = selector.select(population, k_offspring, generation=generation)
            crossover_op = self._choose_operator(self.crossover_ops, self.crossover_weights)
            mutation_op = self._choose_operator(self.mutation_ops, self.mutation_weights)

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

            population = self.survival.apply(population, children, selector)

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

            # Criterio: error minimo alcanzado
            if config.min_error > 0:
                best = population.best
                current_error = (1.0 / best.fitness) - 1.0 if best.fitness > 0 else float('inf')
                if current_error <= config.min_error:
                    stop_reason = "error_minimo"
                    logger.info(
                        "Error minimo alcanzado en generacion %d: error=%.6f <= min_error=%.6f",
                        generation, current_error, config.min_error,
                    )
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
