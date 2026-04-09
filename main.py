import logging
import sys

import numpy as np
from PIL import Image

from config.config_loader import load_config, Config
from core.ga_context import GAContext
from core.genetic_algorithm import GeneticAlgorithm
from core.island_ga import IslandGeneticAlgorithm
from render.cpu_renderer import CPURenderer

logger = logging.getLogger(__name__)

from selection.selection_strategy import SelectionStrategy
from selection.elite import EliteSelection
from selection.roulette import RouletteSelection
from selection.universal import UniversalSelection
from selection.boltzmann import BoltzmannSelection
from selection.tournament import DeterministicTournamentSelection, ProbabilisticTournamentSelection
from selection.ranking import RankingSelection

from crossover.one_point import OnePointCrossover
from crossover.two_point import TwoPointCrossover
from crossover.uniform import UniformCrossover
from crossover.annular import AnnularCrossover
from crossover.arithmetic import ArithmeticCrossover

from mutation.gen_mutation import GenMutation
from mutation.multigen_mutation import MultiGenMutation
from mutation.uniform_mutation import UniformMutation
from mutation.complete_mutation import CompleteMutation
from mutation.non_uniform_mutation import NonUniformMutation
from mutation.gaussian_mutation import GaussianMutation

from survival.additive import AdditiveSurvival
from survival.exclusive import ExclusiveSurvival

from fitness.mse import MSEFitness
from fitness.mae import MAEFitness
from fitness.gmsd import GMSDFitness
from fitness.oklab import OklabMSEFitness
from fitness.msssim import MSSSIMFitness
from fitness.fsim import FSIMFitness
from fitness.ssim import SSIMFitness
from fitness.linear_mse import LinearMSEFitness
from fitness.linear_oklab import LinearOklabFitness


SELECTIONS = {
    "Elite": EliteSelection,
    "Ruleta": RouletteSelection,
    "Universal": UniversalSelection,
    "Boltzmann": BoltzmannSelection,
    "TorneosDeterministicos": DeterministicTournamentSelection,
    "TorneosProbabilisticos": ProbabilisticTournamentSelection,
    "Ranking": RankingSelection,
}

CROSSOVERS = {
    "OnePoint": OnePointCrossover,
    "TwoPoint": TwoPointCrossover,
    "Uniform": UniformCrossover,
    "Annular": AnnularCrossover,
    "Aritmetico": ArithmeticCrossover,
}

MUTATIONS = {
    "Gen": GenMutation,
    "MultiGen": MultiGenMutation,
    "Uniforme": UniformMutation,
    "Completa": CompleteMutation,
    "NoUniforme": NonUniformMutation,
    "Gaussiana": GaussianMutation,
}

SURVIVALS = {
    "Aditiva": AdditiveSurvival,
    "Exclusiva": ExclusiveSurvival,
}

FITNESS = {
    "MSE": MSEFitness,
    "MAE": MAEFitness,
    "GMSD": GMSDFitness,
    "Oklab": OklabMSEFitness,
    "MSSSIM": MSSSIMFitness,
    "FSIM": FSIMFitness,
    "SSIM": SSIMFitness,
    "LinearMSE": LinearMSEFitness,
    "LinearOklab": LinearOklabFitness,
}


def _build_selection(sel_name: str, config: Config) -> SelectionStrategy:
    """Instancia un operador de seleccion por nombre."""
    if sel_name not in SELECTIONS:
        raise ValueError(f"Metodo de seleccion desconocido: '{sel_name}'. Opciones: {list(SELECTIONS.keys())}")
    sel_class = SELECTIONS[sel_name]
    if sel_name == "Boltzmann":
        return sel_class(
            t0=config.boltzmann_t0,
            tc=config.boltzmann_tc,
            k=config.boltzmann_k,
        )
    elif sel_name == "TorneosDeterministicos":
        return sel_class(m=config.tournament_m)
    elif sel_name == "TorneosProbabilisticos":
        return sel_class(threshold=config.tournament_threshold)
    else:
        return sel_class()


def build_operators(config: Config):
    """Instancia todos los operadores a partir de la configuracion.

    Soporta multiples operadores de seleccion con pesos ponderados.
    Retorna (selection_ops, crossover_ops, mutation_ops, survival, fitness)
    donde selection_ops es una lista de operadores de seleccion.
    """

    # Seleccion: soportar multiples metodos via selection_methods
    sel_names = config.selection_methods if config.selection_methods else [config.selection_method]
    selection_ops = [_build_selection(name, config) for name in sel_names]

    # Crossover
    crossover_ops = []
    for cx_name in config.crossover_methods:
        if cx_name not in CROSSOVERS:
            raise ValueError(f"Metodo de crossover desconocido: '{cx_name}'. Opciones: {list(CROSSOVERS.keys())}")
        if cx_name == "Aritmetico":
            crossover_ops.append(CROSSOVERS[cx_name](alpha=config.arithmetic_alpha))
        else:
            crossover_ops.append(CROSSOVERS[cx_name]())

    # Mutacion
    mutation_ops = []
    for mut_name in config.mutation_methods:
        if mut_name not in MUTATIONS:
            raise ValueError(f"Metodo de mutacion desconocido: '{mut_name}'. Opciones: {list(MUTATIONS.keys())}")
        mut_class = MUTATIONS[mut_name]
        if mut_name == "NoUniforme":
            mutation_ops.append(mut_class(mutation_rate=config.mutation_rate, b=config.non_uniform_b))
        elif mut_name == "MultiGen":
            mutation_ops.append(mut_class(mutation_rate=config.mutation_rate, max_genes=config.triangle_count))
        elif mut_name == "Gaussiana":
            mutation_ops.append(mut_class(
                mutation_rate=config.mutation_rate,
                sigma=config.gaussian_sigma,
                sigma_color=config.gaussian_sigma_color,
                decay_b=config.gaussian_decay_b,
                swap_rate=config.gaussian_swap_rate,
            ))
        else:
            mutation_ops.append(mut_class(mutation_rate=config.mutation_rate))

    # Supervivencia
    surv_name = config.survival_strategy
    if surv_name not in SURVIVALS:
        raise ValueError(f"Estrategia de supervivencia desconocida: '{surv_name}'. Opciones: {list(SURVIVALS.keys())}")
    survival = SURVIVALS[surv_name]()

    # Fitness
    fit_name = config.fitness_function
    if fit_name not in FITNESS:
        raise ValueError(f"Funcion de fitness desconocida: '{fit_name}'. Opciones: {list(FITNESS.keys())}")
    fitness = FITNESS[fit_name]()

    return selection_ops, crossover_ops, mutation_ops, survival, fitness


def create_renderer(config: Config, target_image: np.ndarray, width: int, height: int):
    """Crea el renderer apropiado segun la configuracion y disponibilidad."""
    if config.use_gpu:
        from render.gpu_renderer import GPURenderer, gpu_available

        if gpu_available(config.gpu_device):
            device_info = GPURenderer.detect_device(config.gpu_device)
            logger.info("Usando renderer GPU: %s", device_info)
            return GPURenderer(
                width,
                height,
                target_image,
                device_preference=config.gpu_device,
            )

        if config.gpu_device == "dedicated":
            detected = GPURenderer.detect_device("auto")
            raise RuntimeError(
                "GPU dedicada solicitada pero no disponible. "
                f"Dispositivo detectado: {detected}"
            )

        logger.warning(
            "GPU solicitada pero no disponible para preferencia '%s'. Fallback a CPU.",
            config.gpu_device,
        )
        return CPURenderer()

    return CPURenderer()


def _build_island(
    base_config: Config,
    target_image: np.ndarray,
    width: int,
    height: int,
    island_index: int,
    overrides: dict | None = None,
) -> GeneticAlgorithm:
    """Crea una instancia de GeneticAlgorithm para una isla."""
    if overrides:
        from config.config_loader import apply_island_overrides
        config = apply_island_overrides(base_config, overrides)
        name = overrides.get("name", f"island_{island_index}")
        logger.info("Isla %d (%s): config con overrides %s", island_index, name, list(overrides.keys()))
    else:
        config = base_config
        name = f"island_{island_index}"

    renderer = create_renderer(config, target_image, width, height)
    selection_ops, crossover_ops, mutation_ops, survival, fitness = build_operators(config)
    context = GAContext(generation=0, max_generations=config.max_generations)

    for sel in selection_ops:
        if hasattr(sel, 'context'):
            sel.context = context
    for mut in mutation_ops:
        if hasattr(mut, 'context'):
            mut.context = context

    return GeneticAlgorithm(
        config=config,
        target_image=target_image,
        renderer=renderer,
        fitness_fn=fitness,
        selection_ops=selection_ops,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
        context=context,
        output_dir=f"output/{name}",
    )


def run_from_paths(image_path: str, config_path: str):
    """Carga inputs, ejecuta el GA y guarda los outputs principales."""
    config = load_config(config_path)

    img = Image.open(image_path).convert("RGBA")
    target = np.array(img, dtype=np.float32) / 255.0

    height, width = target.shape[0], target.shape[1]

    if config.island_enabled:
        island_overrides = config.island_configs if config.island_configs else [None] * config.island_count
        islands = [
            _build_island(config, target, width, height, i, overrides=island_overrides[i])
            for i in range(config.island_count)
        ]
        island_ga = IslandGeneticAlgorithm(
            islands=islands,
            config=config,
            target_image=target,
        )
        result = island_ga.run()
        logging.info(
            "Island Model finalizado. Output en output/island_*/. "
            "Mejor isla guardada con resultado final."
        )
    else:
        renderer = create_renderer(config, target, width, height)
        selection_ops, crossover_ops, mutation_ops, survival, fitness = build_operators(config)
        context = GAContext(generation=0, max_generations=config.max_generations)

        for sel in selection_ops:
            if hasattr(sel, 'context'):
                sel.context = context
        for mut in mutation_ops:
            if hasattr(mut, 'context'):
                mut.context = context

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
        result = ga.run()
        logging.info("Output guardado en output/final.png y output/triangles.json")

    return result


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI del proyecto."""
    args = sys.argv[1:] if argv is None else argv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    if len(args) < 2:
        print("Uso: python main.py <image_path> <config_path>")
        return 1

    run_from_paths(args[0], args[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
