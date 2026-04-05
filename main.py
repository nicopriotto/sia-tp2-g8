import logging
import sys

import numpy as np
from PIL import Image

from config.config_loader import load_config
from core.genetic_algorithm import GeneticAlgorithm
from crossover.one_point import OnePointCrossover
from fitness.mse import MSEFitness
from mutation.gen_mutation import GenMutation
from render.cpu_renderer import CPURenderer
from selection.elite import EliteSelection
from survival.additive import AdditiveSurvival


def run_from_paths(image_path: str, config_path: str):
    """Carga inputs, ejecuta el GA y guarda los outputs principales."""
    config = load_config(config_path)

    img = Image.open(image_path).convert("RGBA")
    target = np.array(img, dtype=np.float32) / 255.0

    renderer = CPURenderer()
    fitness_fn = MSEFitness()
    selection = EliteSelection()
    crossover_ops = [OnePointCrossover()]
    mutation_ops = [GenMutation(config.mutation_rate)]
    survival = AdditiveSurvival()

    ga = GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=fitness_fn,
        selection=selection,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
    )

    population = ga.run()

    logging.info("Output guardado en output/final.png y output/triangles.json")
    return population


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
