import csv
import json
import os

import numpy as np
from PIL import Image

from config.config_loader import Config
from core.individual import Individual
from core.population import Population


class MetricsCollector:
    def __init__(self, output_dir: str, save_every: int, renderer, width: int, height: int, gene_type: str = "triangle"):
        self.output_dir = output_dir
        self.save_every = save_every
        self.renderer = renderer
        self.width = width
        self.height = height
        self.gene_type = gene_type

    def init_csv(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f"{self.output_dir}/metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness", "avg_fitness", "fitness_std", "elapsed_seconds"])

    def log_generation(self, generation: int, population: Population, elapsed_seconds: float) -> None:
        with open(f"{self.output_dir}/metrics.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                population.best.fitness,
                population.average_fitness,
                population.fitness_std,
                round(elapsed_seconds, 3),
            ])

    def save_snapshot(self, generation: int, best_individual: Individual) -> None:
        if self.save_every <= 0:
            return
        if generation % self.save_every != 0:
            return
        image = self.renderer.render(best_individual.genes, self.width, self.height, gene_type=self.gene_type)
        img_uint8 = (image * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(f"{self.output_dir}/gen_{generation:04d}.png")

    def save_final_result(self, best_individual: Individual, config: Config, total_generations: int) -> None:
        with open(f"{self.output_dir}/triangles.json", "w", encoding="utf-8") as f:
            json.dump(best_individual.to_dict(), f, indent=2)

    def save_final_image(self, best_individual: Individual) -> None:
        image = self.renderer.render(best_individual.genes, self.width, self.height, gene_type=self.gene_type)
        img_uint8 = (image * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(f"{self.output_dir}/final.png")
