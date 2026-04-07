"""
Runner de corridas masivas para presentacion del TP.

Ejecuta campañas separadas por familia de operadores:
- seleccion
- supervivencia
- crossover
- mutacion

Cada corrida se ejecuta para 3 imagenes objetivo y guarda resultados en:
output/batches/<batch_id>/<image_name>/<campaign>/<config_slug>/run_seed_<seed>/
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config_loader import (
    VALID_CROSSOVER_METHODS,
    VALID_FITNESS_FUNCTIONS,
    VALID_MUTATION_METHODS,
    VALID_SELECTION_METHODS,
    VALID_SURVIVAL_STRATEGIES,
)
from experiments.run_experiment import run_single

logger = logging.getLogger(__name__)


FORCED_GLOBAL_OVERRIDES = {
    "max_generations": 15000,
    "fitness_function": "LinearMSE",
    "save_every": 100,
    "max_seconds": 0.0,
    "fitness_threshold": 2.0,
    "content_threshold": 0.0,
    "content_generations": 0,
    "structure_threshold": 0.0,
    "structure_generations": 0,
    "min_error": 0.0,
}


@dataclass(frozen=True)
class RunSpec:
    campaign: str
    slug: str
    description: str
    overrides: dict


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "item"


def make_combo_slug(campaign: str, kind: str, methods: list[str]) -> str:
    method_part = "-".join(slugify(m) for m in methods)
    raw = f"{campaign}-{kind}-{method_part}"
    if len(raw) <= 110:
        return raw
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{campaign}-{kind}-{digest}"


def uniform_weights(count: int) -> list[float]:
    if count <= 0:
        return []
    return [1.0 / count] * count


def _default_selection_params() -> dict:
    return {
        "boltzmann_t0": 100.0,
        "boltzmann_tc": 1.0,
        "boltzmann_k": 0.01,
        "tournament_m": 5,
        "tournament_threshold": 0.75,
    }


def _default_crossover_params() -> dict:
    return {
        "arithmetic_alpha": 0.5,
    }


def _default_mutation_params() -> dict:
    return {
        "non_uniform_b": 1.0,
        "gaussian_sigma": 0.1,
        "gaussian_sigma_color": 0.1,
        "gaussian_decay_b": 0.0,
        "gaussian_swap_rate": 0.0,
    }


def _build_method_variants(methods: list[str], triple_count: int) -> list[tuple[str, list[str]]]:
    variants: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()

    def _add(kind: str, combo: tuple[str, ...]):
        if combo in seen:
            return
        seen.add(combo)
        variants.append((kind, list(combo)))

    for method in methods:
        _add("single", (method,))

    for pair in combinations(methods, 2):
        _add("pair", pair)

    triple_candidates = list(combinations(methods, 3))
    for triple in triple_candidates[:max(0, triple_count)]:
        _add("triple", triple)

    _add("all", tuple(methods))
    return variants


def build_selection_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    variants = _build_method_variants(VALID_SELECTION_METHODS, triple_count=7)
    for kind, methods in variants:
        specs.append(
            RunSpec(
                campaign="selection",
                slug=make_combo_slug("selection", kind, methods),
                description=f"selection methods {methods}",
                overrides={
                    **_default_selection_params(),
                    "selection_method": methods[0],
                    "selection_methods": methods,
                    "selection_weights": uniform_weights(len(methods)),
                },
            )
        )
    return specs


def build_survival_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for strategy in VALID_SURVIVAL_STRATEGIES:
        specs.append(
            RunSpec(
                campaign="survival",
                slug=f"survival-single-{slugify(strategy)}",
                description=f"survival strategy {strategy}",
                overrides={"survival_strategy": strategy},
            )
        )
    return specs


def build_crossover_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    variants = _build_method_variants(VALID_CROSSOVER_METHODS, triple_count=4)
    for kind, methods in variants:
        specs.append(
            RunSpec(
                campaign="crossover",
                slug=make_combo_slug("crossover", kind, methods),
                description=f"crossover methods {methods}",
                overrides={
                    **_default_crossover_params(),
                    "crossover_methods": methods,
                    "crossover_weights": uniform_weights(len(methods)),
                },
            )
        )
    return specs


def build_mutation_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    variants = _build_method_variants(VALID_MUTATION_METHODS, triple_count=6)
    for kind, methods in variants:
        specs.append(
            RunSpec(
                campaign="mutation",
                slug=make_combo_slug("mutation", kind, methods),
                description=f"mutation methods {methods}",
                overrides={
                    **_default_mutation_params(),
                    "mutation_methods": methods,
                    "mutation_weights": uniform_weights(len(methods)),
                },
            )
        )
    return specs


def build_run_specs(campaigns: set[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if "selection" in campaigns:
        specs.extend(build_selection_specs())
    if "survival" in campaigns:
        specs.extend(build_survival_specs())
    if "crossover" in campaigns:
        specs.extend(build_crossover_specs())
    if "mutation" in campaigns:
        specs.extend(build_mutation_specs())
    return specs


def _fixed_config_for_campaign(campaign: str) -> dict:
    fixed = {
        **_default_selection_params(),
        **_default_crossover_params(),
        **_default_mutation_params(),
        "selection_method": "Ranking",
        "selection_methods": ["Ranking"],
        "selection_weights": [1.0],
        "crossover_methods": ["Uniform"],
        "crossover_weights": [1.0],
        "mutation_methods": ["Gaussiana"],
        "mutation_weights": [1.0],
        "survival_strategy": "Aditiva",
    }

    # La campaña de supervivencia evalua solo la estrategia de supervivencia
    # con el resto fijo.
    if campaign == "survival":
        return fixed

    return fixed


def load_base_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_effective_config(base_config: dict, spec: RunSpec) -> dict:
    config = dict(base_config)
    config.update(FORCED_GLOBAL_OVERRIDES)
    config.update(_fixed_config_for_campaign(spec.campaign))
    config.update(spec.overrides)
    return config


def write_supported_options(batch_root: Path, config_count: int) -> None:
    payload = {
        "selection_methods": VALID_SELECTION_METHODS,
        "crossover_methods": VALID_CROSSOVER_METHODS,
        "mutation_methods": VALID_MUTATION_METHODS,
        "survival_strategies": VALID_SURVIVAL_STRATEGIES,
        "fitness_functions": VALID_FITNESS_FUNCTIONS,
        "forced_overrides": FORCED_GLOBAL_OVERRIDES,
        "generated_config_count": config_count,
        "notes": "selection/crossover/mutation incluyen individuales, pares, subset de triples y combinacion total.",
    }
    out_path = batch_root / "supported_config_options.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_batch(
    images: list[Path],
    base_config_path: Path,
    output_root: Path,
    seed: int,
    campaigns: set[str],
    dry_run: bool = False,
) -> int:
    base_config = load_base_config(base_config_path)
    specs = build_run_specs(campaigns)
    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_root = output_root / batch_id
    batch_root.mkdir(parents=True, exist_ok=True)

    write_supported_options(batch_root, len(specs))

    total_runs = len(images) * len(specs)
    logger.info("Batch %s", batch_id)
    logger.info("Imagenes: %d | Configuraciones: %d | Corridas totales: %d", len(images), len(specs), total_runs)

    manifest_path = batch_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "campaign",
            "config_slug",
            "description",
            "seed",
            "status",
            "stop_reason",
            "final_fitness",
            "final_generation",
            "elapsed_seconds",
            "run_path",
            "error",
        ])

        run_index = 0
        for image_path in images:
            image_label = slugify(image_path.stem)
            for spec in specs:
                run_index += 1
                config = build_effective_config(base_config, spec)
                run_dir = (
                    batch_root
                    / image_label
                    / spec.campaign
                    / spec.slug
                    / f"run_seed_{seed:04d}"
                )

                logger.info("[%d/%d] %s | %s | %s", run_index, total_runs, image_label, spec.campaign, spec.slug)

                if dry_run:
                    writer.writerow([
                        str(image_path),
                        spec.campaign,
                        spec.slug,
                        spec.description,
                        seed,
                        "dry_run",
                        "",
                        "",
                        "",
                        "",
                        str(run_dir),
                        "",
                    ])
                    continue

                status = "ok"
                stop_reason = ""
                final_fitness = ""
                final_generation = ""
                elapsed_seconds = ""
                error_msg = ""

                try:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    result = run_single(
                        config_dict=config,
                        image_path=str(image_path),
                        output_dir=str(run_dir),
                        seed=seed,
                    )
                    stop_reason = result.stop_reason
                    final_fitness = f"{result.best_individual.fitness:.8f}"
                    final_generation = result.final_generation
                    elapsed_seconds = f"{result.elapsed_seconds:.3f}"
                except Exception as exc:
                    status = "error"
                    error_msg = str(exc)
                    logger.exception("Fallo corrida %s/%s/%s", image_label, spec.campaign, spec.slug)
                    with (run_dir / "error.txt").open("w", encoding="utf-8") as ef:
                        ef.write(error_msg)

                writer.writerow([
                    str(image_path),
                    spec.campaign,
                    spec.slug,
                    spec.description,
                    seed,
                    status,
                    stop_reason,
                    final_fitness,
                    final_generation,
                    elapsed_seconds,
                    str(run_dir),
                    error_msg,
                ])

    logger.info("Batch finalizado. Manifest: %s", manifest_path)
    return 0


def parse_campaigns(raw: str) -> set[str]:
    if not raw:
        return {"selection", "survival", "crossover", "mutation"}
    campaigns = {slugify(part).replace("-", "_") for part in raw.split(",")}
    valid = {"selection", "survival", "crossover", "mutation"}
    invalid = campaigns - valid
    if invalid:
        raise ValueError(f"Campañas invalidas: {sorted(invalid)}. Opciones: {sorted(valid)}")
    return campaigns


def validate_images(images: list[str]) -> list[Path]:
    if len(images) != 3:
        raise ValueError("Debes pasar exactamente 3 imagenes con --images.")
    resolved: list[Path] = []
    for img in images:
        path = Path(img).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No existe la imagen: {path}")
        if not path.is_file():
            raise ValueError(f"La ruta no es un archivo: {path}")
        resolved.append(path)
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta un batch grande de corridas por seleccion/supervivencia/crossover/mutacion sobre 3 imagenes."
    )
    parser.add_argument(
        "--images",
        nargs=3,
        required=True,
        help="Rutas de exactamente 3 imagenes objetivo.",
    )
    parser.add_argument(
        "--base-config",
        default=os.path.join(PROJECT_ROOT, "run_configs", "config.json"),
        help="Config base JSON para completar parametros no barridos.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(PROJECT_ROOT, "output", "batches"),
        help="Directorio raiz donde se creara el batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed unica para todas las corridas.",
    )
    parser.add_argument(
        "--campaigns",
        default="selection,survival,crossover,mutation",
        help="Campañas separadas por coma. Opciones: selection,survival,crossover,mutation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No ejecuta GA; solo genera manifest y estructura de corrida.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    images = validate_images(args.images)
    base_config_path = Path(args.base_config).expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"No existe base-config: {base_config_path}")

    campaigns = parse_campaigns(args.campaigns)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    return run_batch(
        images=images,
        base_config_path=base_config_path,
        output_root=output_root,
        seed=args.seed,
        campaigns=campaigns,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
