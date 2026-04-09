import logging
import time

import numpy as np

from config.config_loader import Config
from core.ga_result import GAResult
from core.genetic_algorithm import EpochState, GeneticAlgorithm
from core.individual import Individual
from core.metrics_collector import MetricsCollector
from core.population import Population

logger = logging.getLogger(__name__)


class IslandGeneticAlgorithm:
    """Orquestador de multiples GeneticAlgorithm (islas) con migracion periodica."""

    def __init__(
        self,
        islands: list[GeneticAlgorithm],
        config: Config,
        target_image: np.ndarray,
    ):
        self.islands = islands
        self.config = config
        self.target_image = target_image
        self.num_islands = len(islands)

    def run(self) -> GAResult:
        """Ejecuta el island model completo."""
        config = self.config
        interval = config.island_migration_interval
        max_gen = config.max_generations
        height, width = self.target_image.shape[:2]

        start_time = time.time()

        # Inicializar poblaciones, collectors y estados por isla
        states: list[EpochState] = []
        collectors: list[MetricsCollector] = []
        gene_type = getattr(config, 'gene_type', 'triangle')

        for i, island in enumerate(self.islands):
            island_config = island.config
            output_dir = island.output_dir
            collector = MetricsCollector(
                output_dir=output_dir,
                save_every=island_config.save_every,
                renderer=island.renderer,
                width=width,
                height=height,
                gene_type=gene_type,
            )
            collector.init_csv()
            collectors.append(collector)

            if getattr(island_config, 'grid_init', False):
                pop = Population.grid_init(
                    island_config.population_size, island_config.triangle_count,
                    self.target_image,
                )
            elif getattr(island_config, 'smart_init', False):
                pop = Population.smart_random(
                    island_config.population_size, island_config.triangle_count,
                    gene_type, self.target_image,
                )
            else:
                pop = Population.random(
                    island_config.population_size, island_config.triangle_count, gene_type,
                )
            pop.evaluate_all(self.target_image, island.renderer, island.fitness_fn)

            init_elapsed = time.time() - start_time
            collector.log_generation(0, pop, init_elapsed, generation_seconds=init_elapsed)
            collector.save_snapshot(0, pop.best)

            state = EpochState(
                population=pop,
                best_fitness_history=[pop.best.fitness],
            )
            states.append(state)

            logger.info(
                "Isla %d | Poblacion inicial | Best: %.4f",
                i, pop.best.fitness,
            )

        # Loop principal por epochs
        current_gen = 1
        global_stop_reason = None
        best_fitness_history: list[float] = []

        while current_gen <= max_gen:
            epoch_end = min(current_gen + interval, max_gen + 1)

            # Correr cada isla por un epoch
            for i, island in enumerate(self.islands):
                states[i] = island.run_epoch(
                    states[i], current_gen, epoch_end,
                    start_time, collectors[i],
                )
                if states[i].stop_reason is not None:
                    global_stop_reason = f"isla_{i}:{states[i].stop_reason}"

            # Registrar mejor fitness global del epoch
            global_best = max(
                (s.population.best for s in states),
                key=lambda ind: ind.fitness,
            )
            best_fitness_history.append(global_best.fitness)

            logger.info(
                "Epoch [gen %d-%d] | Mejor global: %.4f",
                current_gen, epoch_end - 1, global_best.fitness,
            )

            if global_stop_reason is not None:
                break

            # Migracion
            self._migrate([s.population for s in states])

            current_gen = epoch_end

        if global_stop_reason is None:
            global_stop_reason = "generaciones_maximas"

        # Encontrar la mejor isla
        best_island_idx = max(
            range(self.num_islands),
            key=lambda i: states[i].population.best.fitness,
        )
        best_individual = states[best_island_idx].population.best

        elapsed = time.time() - start_time

        # Guardar resultado final de la mejor isla
        best_collector = collectors[best_island_idx]
        best_collector.save_final_result(best_individual, config, current_gen - 1)
        best_collector.save_final_image(best_individual)

        logger.info("=== FIN ISLAND MODEL ===")
        logger.info("Mejor isla: %d", best_island_idx)
        logger.info("Motivo de corte: %s", global_stop_reason)
        logger.info("Mejor fitness final: %.4f", best_individual.fitness)

        return GAResult(
            best_individual=best_individual,
            final_generation=current_gen - 1,
            stop_reason=global_stop_reason,
            elapsed_seconds=elapsed,
            best_fitness_history=best_fitness_history,
        )

    def _migrate(self, populations: list[Population]) -> None:
        """Ejecuta migracion entre islas segun la topologia configurada."""
        topology = self.config.island_topology
        count = self.config.island_migration_count

        if topology == "ring":
            self._migrate_ring(populations, count)
        elif topology == "fully_connected":
            self._migrate_fully_connected(populations, count)

    def _migrate_ring(self, populations: list[Population], count: int) -> None:
        """Topologia anillo: isla i envia sus mejores a isla (i+1) % N."""
        n = len(populations)
        # Recolectar emigrantes antes de modificar poblaciones
        emigrants: list[list[Individual]] = []
        for pop in populations:
            sorted_inds = sorted(pop.individuals, key=lambda x: x.fitness, reverse=True)
            emigrants.append([ind.copy() for ind in sorted_inds[:count]])

        for i in range(n):
            dest = (i + 1) % n
            dest_pop = populations[dest]
            # Reemplazar los peores de la isla destino
            dest_pop.individuals.sort(key=lambda x: x.fitness)
            for j, emigrant in enumerate(emigrants[i]):
                if j < len(dest_pop.individuals):
                    dest_pop.individuals[j] = emigrant

        logger.info("Migracion ring: %d individuos por isla", count)

    def _migrate_fully_connected(self, populations: list[Population], count: int) -> None:
        """Topologia fully connected: cada isla envia sus mejores a todas las demas."""
        n = len(populations)
        # Recolectar emigrantes
        emigrants: list[list[Individual]] = []
        for pop in populations:
            sorted_inds = sorted(pop.individuals, key=lambda x: x.fitness, reverse=True)
            emigrants.append([ind.copy() for ind in sorted_inds[:count]])

        for dest in range(n):
            # Recibir de todas las otras islas
            incoming: list[Individual] = []
            for src in range(n):
                if src != dest:
                    incoming.extend([ind.copy() for ind in emigrants[src]])

            # Reemplazar los peores
            dest_pop = populations[dest]
            dest_pop.individuals.sort(key=lambda x: x.fitness)
            replacements = min(len(incoming), len(dest_pop.individuals) - 1)
            for j in range(replacements):
                if incoming[j].fitness > dest_pop.individuals[j].fitness:
                    dest_pop.individuals[j] = incoming[j]

        logger.info("Migracion fully_connected: %d individuos por isla", count)
