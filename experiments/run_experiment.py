"""
Script base para ejecutar un experimento con multiples configuraciones.

Provee utilidades reutilizables: configuracion base, seeds fijas,
funcion run_single para ejecutar una corrida del GA, y run_experiment
para barrer un hiperparametro.
"""
import json
import os
import random
import sys
import logging

import numpy as np
from PIL import Image

# Agregar el directorio raiz del proyecto al path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config_loader import Config
from core.ga_context import GAContext
from core.genetic_algorithm import GeneticAlgorithm
from main import build_operators, create_renderer

logger = logging.getLogger(__name__)

# Seeds fijas para reproducibilidad
SEEDS = [42, 123, 456, 789, 1024]

# Configuracion base por defecto para todos los experimentos
BASE_CONFIG = {
    "triangle_count": 100,
    "population_size": 50,
    "max_generations": 1000,
    "fitness_threshold": 0.9999,
    "selection_method": "Boltzmann",
    "crossover_methods": ["Uniform"],
    "crossover_probability": 0.7,
    "mutation_methods": ["Gaussiana"],
    "mutation_rate": 0.05,
    "survival_strategy": "Aditiva",
    "fitness_function": "LinearMSE",
    "k_offspring": 50,
    "save_every": 50,
    "smart_init": True,
    "use_gpu": True,
    "gpu_device": "dedicated",
    "gaussian_sigma": 0.1,
    "gaussian_sigma_color": 0.08,
    "gaussian_decay_b": 2.0,
    "gaussian_swap_rate": 0.05,
    "elite_count": 2,
    "boltzmann_t0": 100.0,
    "boltzmann_tc": 1.0,
    "boltzmann_k": 0.01,
}

# Seeds reducidas para presentacion (3 seeds = rapido + estadisticamente util)
SEEDS = [42, 123, 456]

# Imagen por defecto: Grecia (complejidad media)
IMAGE_PATH = os.path.join(PROJECT_ROOT, "images", "2.jpg")

# Las 4 imagenes principales para correr todos los tests
ALL_IMAGES = [
    ("1_Ucrania", os.path.join(PROJECT_ROOT, "images", "1.jpg")),
    ("2_Grecia", os.path.join(PROJECT_ROOT, "images", "2.jpg")),
    ("3_Apple", os.path.join(PROJECT_ROOT, "images", "3.jpg")),
    ("4_Cubista", os.path.join(PROJECT_ROOT, "images", "4.jpg")),
]


def load_target_image(image_path: str) -> np.ndarray:
    """Carga la imagen objetivo como numpy array float32 RGBA en [0, 1]."""
    img = Image.open(image_path).convert("RGBA")
    return np.array(img, dtype=np.float32) / 255.0


def run_single(config_dict: dict, image_path: str, output_dir: str, seed: int):
    """
    Ejecuta una corrida del GA con configuracion y seed dados.

    Parametros:
        config_dict: diccionario con los campos de Config
        image_path: ruta a la imagen objetivo
        output_dir: directorio donde guardar resultados (metrics.csv, imagenes)
        seed: semilla para reproducibilidad
    """
    # Fijar seeds para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Guardar la configuracion usada
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({**config_dict, "seed": seed}, f, indent=2)

    # Crear Config y operadores
    config = Config(**config_dict)
    target = load_target_image(image_path)
    height, width = target.shape[0], target.shape[1]

    renderer = create_renderer(config, target, width, height)
    selection_ops, crossover_ops, mutation_ops, survival, fitness_fn = build_operators(config)

    context = GAContext(generation=0, max_generations=config.max_generations)

    # Inyectar contexto a operadores que lo necesiten
    for sel in selection_ops:
        if hasattr(sel, "context"):
            sel.context = context
    for mut in mutation_ops:
        if hasattr(mut, "context"):
            mut.context = context

    ga = GeneticAlgorithm(
        config=config,
        target_image=target,
        renderer=renderer,
        fitness_fn=fitness_fn,
        selection_ops=selection_ops,
        crossover_ops=crossover_ops,
        mutation_ops=mutation_ops,
        survival=survival,
        context=context,
        output_dir=output_dir,
    )

    result = ga.run()
    logger.info(
        "Corrida finalizada: fitness=%.4f, generaciones=%d, motivo=%s",
        result.best_individual.fitness,
        result.final_generation,
        result.stop_reason,
    )
    return result


def run_experiment(
    name: str,
    configs: list[tuple[str, dict]],
    output_base: str,
    image_path: str | None = None,
    seeds: list[int] | None = None,
    max_generations: int | None = None,
):
    """
    Ejecuta multiples corridas variando configuraciones.

    Parametros:
        name: nombre del experimento (para logging)
        configs: lista de (etiqueta, overrides_dict) donde overrides se
                 aplican sobre BASE_CONFIG
        output_base: directorio base de resultados
        image_path: ruta a la imagen objetivo (default: IMAGE_PATH)
        seeds: lista de seeds a usar (default: SEEDS)
        max_generations: override de generaciones maximas (util para --quick)
    """
    if image_path is None:
        image_path = IMAGE_PATH
    if seeds is None:
        seeds = SEEDS

    for label, overrides in configs:
        config_dict = {**BASE_CONFIG, **overrides}
        if max_generations is not None:
            config_dict["max_generations"] = max_generations

        for seed in seeds:
            run_dir = os.path.join(output_base, label, f"seed_{seed}")
            print(f"[{name}] {label}, seed={seed} -> {run_dir}")
            try:
                run_single(config_dict, image_path, run_dir, seed)
            except Exception as e:
                print(f"  ERROR: {e}")
                logger.exception("Error en corrida %s seed=%d", label, seed)


def run_experiment_all_images(
    name: str,
    configs: list[tuple[str, dict]],
    output_base: str,
    images: list[tuple[str, str]] | None = None,
    seeds: list[int] | None = None,
    max_generations: int | None = None,
):
    """
    Ejecuta multiples corridas variando configuraciones en TODAS las imagenes.

    Estructura de salida:
        output_base/<imagen>/<config>/<seed>/metrics.csv

    Parametros:
        name: nombre del experimento (para logging)
        configs: lista de (etiqueta, overrides_dict)
        output_base: directorio base de resultados
        images: lista de (etiqueta, ruta) (default: ALL_IMAGES)
        seeds: lista de seeds a usar (default: SEEDS)
        max_generations: override de generaciones maximas
    """
    if images is None:
        images = ALL_IMAGES
    if seeds is None:
        seeds = SEEDS

    for img_label, image_path in images:
        for label, overrides in configs:
            config_dict = {**BASE_CONFIG, **overrides}
            if max_generations is not None:
                config_dict["max_generations"] = max_generations

            for seed in seeds:
                run_dir = os.path.join(output_base, img_label, label, f"seed_{seed}")
                print(f"[{name}] {img_label}/{label}, seed={seed} -> {run_dir}")
                try:
                    run_single(config_dict, image_path, run_dir, seed)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    logger.exception("Error en corrida %s/%s seed=%d", img_label, label, seed)


def parse_common_args(description: str):
    """Parser de argumentos comunes para todos los scripts de experimento."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--max_generations", type=int, default=None,
        help="Override de generaciones maximas (default: usar BASE_CONFIG)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Modo rapido: 50 generaciones, 1 seed"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Seeds a usar (default: todas)"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Ruta a la imagen objetivo"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directorio de salida"
    )
    return parser


def apply_common_args(args):
    """Procesa argumentos comunes y retorna (max_generations, seeds, image_path)."""
    max_gen = args.max_generations
    seeds = args.seeds

    if args.quick:
        max_gen = max_gen or 50
        seeds = seeds or [42]

    image_path = args.image or IMAGE_PATH
    return max_gen, seeds, image_path
