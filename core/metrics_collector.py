import csv
import json
import os

import numpy as np
from PIL import Image

from config.config_loader import Config
from core.individual import Individual
from core.population import Population


class MetricsCollector:
    def __init__(
        self,
        output_dir: str,
        save_every: int,
        renderer,
        width: int,
        height: int,
        gene_type: str = "triangle",
        target_image: np.ndarray | None = None,
    ):
        self.output_dir = output_dir
        self.save_every = save_every
        self.renderer = renderer
        self.width = width
        self.height = height
        self.gene_type = gene_type
        self.target_image = target_image

    @staticmethod
    def _to_uint8_rgba(image: np.ndarray) -> np.ndarray:
        clipped = np.clip(image, 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        clipped = np.clip(image[:, :, :3], 0.0, 1.0)
        return (clipped * 255).astype(np.uint8)

    def init_csv(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f"{self.output_dir}/metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "generation",
                "best_fitness",
                "avg_fitness",
                "fitness_std",
                "elapsed_seconds",
                "generation_seconds",
                "log_line",
            ])

    def log_generation(
        self,
        generation: int,
        population: Population,
        elapsed_seconds: float,
        generation_seconds: float = 0.0,
    ) -> None:
        log_line = (
            f"| Gen {generation} | Best: {population.best.fitness:.4f} | "
            f"Avg: {population.average_fitness:.4f} | "
            f"Std: {population.fitness_std:.4f} | Time: {generation_seconds:.1f}s"
        )
        with open(f"{self.output_dir}/metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                population.best.fitness,
                population.average_fitness,
                population.fitness_std,
                round(elapsed_seconds, 3),
                round(generation_seconds, 3),
                log_line,
            ])

    def save_snapshot(self, generation: int, best_individual: Individual) -> None:
        if self.save_every <= 0:
            return
        if generation % self.save_every != 0:
            return
        image = self.renderer.render(best_individual.genes, self.width, self.height, gene_type=self.gene_type)
        img_uint8 = self._to_uint8_rgb(image)
        Image.fromarray(img_uint8).save(f"{self.output_dir}/gen_{generation:04d}.png")

    def save_final_result(self, best_individual: Individual, config: Config, total_generations: int) -> None:
        with open(f"{self.output_dir}/triangles.json", "w", encoding="utf-8") as f:
            json.dump(best_individual.to_dict(), f, indent=2)

    def save_final_image(self, best_individual: Individual) -> None:
        image = self.renderer.render(best_individual.genes, self.width, self.height, gene_type=self.gene_type)
        img_uint8 = self._to_uint8_rgba(image)
        Image.fromarray(img_uint8).save(f"{self.output_dir}/final.png")

        if self.target_image is None:
            return

        generated_rgb = self._to_uint8_rgb(image)
        target_rgb = self._to_uint8_rgb(self.target_image)

        Image.fromarray(generated_rgb).save(f"{self.output_dir}/final_eval_rgb.png")
        Image.fromarray(target_rgb).save(f"{self.output_dir}/target_eval_rgb.png")

        diff_rgb = np.abs(image[:, :, :3] - self.target_image[:, :, :3])
        diff_rgb_uint8 = self._to_uint8_rgb(diff_rgb)
        Image.fromarray(diff_rgb_uint8).save(f"{self.output_dir}/diff_eval_rgb.png")

        comparison = np.concatenate([target_rgb, generated_rgb, diff_rgb_uint8], axis=1)
        Image.fromarray(comparison).save(f"{self.output_dir}/comparison_eval_rgb.png")
