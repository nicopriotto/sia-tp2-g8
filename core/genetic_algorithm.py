import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

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


@dataclass
class EpochState:
    """Estado acumulado entre epochs de un GeneticAlgorithm."""
    population: Population
    best_fitness_history: list[float] = field(default_factory=list)
    low_diversity_counter: int = 0
    selection_weight_history: list[list[float]] = field(default_factory=list)
    mutation_weight_history: list[list[float]] = field(default_factory=list)
    last_generation: int = 0
    stop_reason: str | None = None


class GeneticAlgorithm:
    """Loop principal del algoritmo genetico."""

    _MIN_ADAPTIVE_WEIGHT = 1e-3

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

        from render.cpu_renderer import CPURenderer
        self._parallel = isinstance(renderer, CPURenderer)

        self.adaptive_operator_weights = config.adaptive_operator_weights
        self.adaptive_operator_delta = config.adaptive_operator_delta

        # Pesos para seleccion ponderada de operadores
        self.selection_weights = config.selection_weights if config.selection_weights else [1.0] * len(self.selection_ops)
        self.crossover_weights = config.crossover_weights if config.crossover_weights else [1.0] * len(self.crossover_ops)
        self.mutation_weights = config.mutation_weights if config.mutation_weights else [1.0] * len(self.mutation_ops)

        if self.adaptive_operator_weights:
            self.selection_weights = self._initial_weights(self.selection_ops)
            self.mutation_weights = self._initial_weights(self.mutation_ops)

    @staticmethod
    def _choose_operator(operators: list, weights: list):
        """Elige un operador segun pesos ponderados."""
        return random.choices(operators, weights=weights, k=1)[0]

    @staticmethod
    def _choose_operator_indexed(operators: list, weights: list) -> tuple[int, object]:
        """Elige un operador segun pesos ponderados y retorna tambien su indice."""
        index = random.choices(range(len(operators)), weights=weights, k=1)[0]
        return index, operators[index]

    @staticmethod
    def _uniform_weights(count: int) -> list[float]:
        if count <= 0:
            return []
        return [1.0 / count] * count

    @classmethod
    def _normalize_weights(cls, weights: list[float], min_weight: float | None = None) -> list[float]:
        if not weights:
            return []

        adjusted = list(weights)
        if min_weight is not None:
            adjusted = [max(min_weight, weight) for weight in adjusted]

        total = sum(adjusted)
        if total <= 0:
            return cls._uniform_weights(len(adjusted))

        return [weight / total for weight in adjusted]

    @classmethod
    def _initial_weights(cls, operators: list) -> list[float]:
        if len(operators) <= 1:
            return [1.0] * len(operators)
        return cls._uniform_weights(len(operators))

    @classmethod
    def _update_weights(
        cls,
        weights: list[float],
        chosen_index: int | None,
        improved: bool,
        delta: float,
    ) -> list[float]:
        if not weights or chosen_index is None:
            return list(weights)

        if len(weights) == 1:
            return [1.0]

        updated = list(weights)
        change = delta if improved else -delta
        updated[chosen_index] = updated[chosen_index] + change
        return cls._normalize_weights(updated, min_weight=cls._MIN_ADAPTIVE_WEIGHT)

    def run(self) -> GAResult:
        """Ejecuta el ciclo evolutivo completo y retorna un GAResult."""
        config = self.config
        height, width = self.target_image.shape[0], self.target_image.shape[1]

        gene_type = getattr(config, 'gene_type', 'triangle')

        collector = MetricsCollector(
            output_dir=self.output_dir,
            save_every=config.save_every,
            renderer=self.renderer,
            width=width,
            height=height,
            gene_type=gene_type,
        )
        collector.init_csv()

        start_time = time.time()

        gene_type = getattr(config, 'gene_type', 'triangle')
        if getattr(config, 'smart_init', False):
            population = Population.smart_random(
                config.population_size, config.triangle_count, gene_type, self.target_image,
            )
        else:
            population = Population.random(config.population_size, config.triangle_count, gene_type)
        population.evaluate_all(self.target_image, self.renderer, self.fitness_fn)

        init_elapsed = time.time() - start_time
        collector.log_generation(0, population, init_elapsed, generation_seconds=init_elapsed)
        collector.save_snapshot(0, population.best)

        logger.info(
            "Poblacion inicial | Best: %.4f | Avg: %.4f | Std: %.4f",
            population.best.fitness,
            population.average_fitness,
            population.fitness_std,
        )

        state = EpochState(
            population=population,
            best_fitness_history=[population.best.fitness],
        )
        if self.adaptive_operator_weights:
            state.selection_weight_history.append(self.selection_weights.copy())
            state.mutation_weight_history.append(self.mutation_weights.copy())

        state = self.run_epoch(state, 1, config.max_generations + 1, start_time, collector)

        if state.stop_reason is None:
            state.stop_reason = "generaciones_maximas"

        elapsed_seconds = time.time() - start_time

        collector.save_final_result(state.population.best, config, state.last_generation)
        collector.save_final_image(state.population.best)

        logger.info("=== FIN ===")
        logger.info("Motivo de corte: %s", state.stop_reason)
        logger.info("Mejor fitness final: %.4f", state.population.best.fitness)
        logger.info("Fitness promedio final: %.4f", state.population.average_fitness)
        logger.info("Desviacion estandar final: %.4f", state.population.fitness_std)

        return GAResult(
            best_individual=state.population.best,
            final_generation=state.last_generation,
            stop_reason=state.stop_reason,
            elapsed_seconds=elapsed_seconds,
            best_fitness_history=state.best_fitness_history,
            selection_weight_history=state.selection_weight_history,
            mutation_weight_history=state.mutation_weight_history,
        )

    def run_epoch(
        self,
        state: EpochState,
        start_gen: int,
        end_gen: int,
        start_time: float,
        collector: MetricsCollector,
    ) -> EpochState:
        """Ejecuta generaciones desde start_gen hasta end_gen (exclusive).

        Recibe y retorna un EpochState con todo el estado acumulado.
        No inicializa poblacion ni guarda resultado final.
        """
        config = self.config
        population = state.population
        best_fitness_history = state.best_fitness_history
        low_diversity_counter = state.low_diversity_counter

        k_offspring = max(2, int(config.generational_gap * config.population_size))
        if k_offspring % 2 != 0:
            k_offspring += 1

        target = self.target_image
        renderer = self.renderer
        fitness_fn = self.fitness_fn

        def _eval_child(child):
            child.compute_fitness(target, renderer, fitness_fn)

        executor = ThreadPoolExecutor() if self._parallel else None

        for generation in range(start_gen, end_gen):
            generation_start = time.time()

            if self.context is not None:
                self.context.generation = generation

            # Criterio: tiempo maximo
            if config.max_seconds > 0:
                elapsed = time.time() - start_time
                if elapsed >= config.max_seconds:
                    state.stop_reason = "tiempo_maximo"
                    state.last_generation = generation - 1
                    break

            previous_best_fitness = population.best.fitness

            # Elitismo: copiar los top-N antes de generar offspring
            elite_count = min(config.elite_count, len(population.individuals) - 1)
            if elite_count > 0:
                sorted_by_fitness = sorted(
                    population.individuals, key=lambda i: i.fitness, reverse=True
                )
                elite = [ind.copy() for ind in sorted_by_fitness[:elite_count]]
            else:
                elite = []

            selection_index, selector = self._choose_operator_indexed(
                self.selection_ops,
                self.selection_weights,
            )
            parents = selector.select(population, k_offspring, generation=generation)
            crossover_op = self._choose_operator(self.crossover_ops, self.crossover_weights)
            mutation_index, mutation_op = self._choose_operator_indexed(
                self.mutation_ops,
                self.mutation_weights,
            )

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

            if executor is not None:
                list(executor.map(_eval_child, children))
            else:
                for child in children:
                    child.compute_fitness(target, renderer, fitness_fn)

            population = self.survival.apply(population, children, selector)

            # Inyectar elite si se perdieron en la supervivencia
            if elite:
                population.individuals.sort(key=lambda i: i.fitness)
                for elite_ind in elite:
                    if elite_ind.fitness > population.individuals[0].fitness:
                        population.individuals[0] = elite_ind
                        population.individuals.sort(key=lambda i: i.fitness)

            if self.adaptive_operator_weights:
                improved = population.best.fitness > previous_best_fitness
                self.selection_weights = self._update_weights(
                    self.selection_weights,
                    selection_index,
                    improved,
                    self.adaptive_operator_delta,
                )
                self.mutation_weights = self._update_weights(
                    self.mutation_weights,
                    mutation_index,
                    improved,
                    self.adaptive_operator_delta,
                )
                state.selection_weight_history.append(self.selection_weights.copy())
                state.mutation_weight_history.append(self.mutation_weights.copy())

            best_fitness_history.append(population.best.fitness)

            collector.save_snapshot(generation, population.best)
            generation_time = time.time() - generation_start
            elapsed = time.time() - start_time
            collector.log_generation(
                generation,
                population,
                elapsed,
                generation_seconds=generation_time,
            )

            logger.info(
                "Gen %d | Best: %.4f | Avg: %.4f | Std: %.4f | Time: %.1fs",
                generation,
                population.best.fitness,
                population.average_fitness,
                population.fitness_std,
                generation_time,
            )

            state.last_generation = generation

            # Criterio: fitness threshold
            if population.best.fitness >= config.fitness_threshold:
                state.stop_reason = "fitness_alcanzado"
                break

            # Criterio: error minimo alcanzado
            if config.min_error > 0:
                best = population.best
                current_error = (1.0 / best.fitness) - 1.0 if best.fitness > 0 else float('inf')
                if current_error <= config.min_error:
                    state.stop_reason = "error_minimo"
                    logger.info(
                        "Error minimo alcanzado en generacion %d: error=%.6f <= min_error=%.6f",
                        generation, current_error, config.min_error,
                    )
                    break

            # Criterio: contenido
            if config.content_generations > 0 and len(best_fitness_history) >= config.content_generations:
                recent = best_fitness_history[-config.content_generations:]
                if max(recent) - min(recent) < config.content_threshold:
                    state.stop_reason = "contenido"
                    break

            # Criterio: estructura
            if config.structure_generations > 0:
                if population.fitness_std < config.structure_threshold:
                    low_diversity_counter += 1
                else:
                    low_diversity_counter = 0
                if low_diversity_counter >= config.structure_generations:
                    state.stop_reason = "estructura"
                    break

            # Criterio: generaciones maximas
            if generation >= config.max_generations:
                state.stop_reason = "generaciones_maximas"
                break

        if executor is not None:
            executor.shutdown(wait=False)

        state.population = population
        state.low_diversity_counter = low_diversity_counter
        return state
