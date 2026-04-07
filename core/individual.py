from dataclasses import dataclass, field

import numpy as np

from genes import gene_layout
from render.renderer import Renderer
from fitness.fitness_function import FitnessFunction


@dataclass
class Individual:
    """Representa un individuo del algoritmo genetico.

    Un individuo es un array numpy de genes (shape n_genes x 11) y su fitness.
    El fitness se inicializa en 0.0 y se actualiza con compute_fitness().
    """
    genes: np.ndarray
    gene_type: str = "triangle"
    fitness: float = 0.0
    fitness_valid: bool = False

    @classmethod
    def random(cls, n_triangles: int, gene_type: str = "triangle") -> "Individual":
        """Crea un individuo con n_triangles genes aleatorios.

        Args:
            n_triangles: Cantidad de genes del individuo.
            gene_type: Tipo de gen a crear ("triangle" o "ellipse").

        Returns:
            Individual con genes aleatorios y fitness 0.0.
        """
        genes = gene_layout.random_genes(gene_type, n_triangles)
        return cls(genes=genes, gene_type=gene_type)

    @classmethod
    def smart_random(cls, n_triangles: int, gene_type: str, target_image: np.ndarray) -> "Individual":
        """Crea un individuo con genes aleatorios y colores de la imagen target."""
        genes = gene_layout.smart_random_genes(gene_type, n_triangles, target_image)
        return cls(genes=genes, gene_type=gene_type)

    def copy(self) -> "Individual":
        """Retorna una copia profunda del individuo.

        El array de genes se copia independientemente.
        El fitness se copia tal cual.
        """
        return Individual(
            genes=self.genes.copy(),
            gene_type=self.gene_type,
            fitness=self.fitness,
            fitness_valid=self.fitness_valid,
        )

    def compute_fitness(
        self,
        target: np.ndarray,
        renderer: Renderer,
        fitness_fn: FitnessFunction,
    ) -> None:
        """Renderiza los genes, calcula el fitness contra el target, y lo almacena.

        Args:
            target: Imagen objetivo, array numpy (H, W, 4), float32, [0, 1].
            renderer: Instancia de Renderer para convertir genes a imagen.
            fitness_fn: Instancia de FitnessFunction para calcular similitud.

        Efecto secundario:
            Actualiza self.fitness con el valor calculado.
        """
        if self.fitness_valid:
            return
        # Intentar path GPU (render + fitness en shaders, sin readback)
        gpu_fitness = renderer.compute_fitness(
            self.genes, fitness_type=fitness_fn.name, gene_type=self.gene_type
        )
        if gpu_fitness is not None:
            self.fitness = gpu_fitness
        else:
            # Path CPU: render + fitness por separado
            height, width = target.shape[0], target.shape[1]
            generated = renderer.render(self.genes, width, height, gene_type=self.gene_type)
            self.fitness = fitness_fn.compute(generated, target)
        self.fitness_valid = True

    def to_dict(self) -> dict:
        """Serializa el individuo a un diccionario.

        Returns:
            dict con claves 'genes' (lista de dicts) y 'fitness' (float).
        """
        return {
            "genes": [
                gene_layout.row_to_dict(self.genes[i], self.gene_type)
                for i in range(self.genes.shape[0])
            ],
            "fitness": float(self.fitness),
        }
