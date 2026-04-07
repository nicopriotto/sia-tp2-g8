from dataclasses import dataclass
import numpy as np
from core.individual import Individual
from render.renderer import Renderer
from fitness.fitness_function import FitnessFunction


@dataclass
class Population:
    """Representa la poblacion del algoritmo genetico.

    Una poblacion es una lista de individuos con propiedades agregadas.
    """
    individuals: list[Individual]

    @classmethod
    def random(cls, size: int, n_triangles: int, gene_type: str = "triangle") -> "Population":
        """Crea una poblacion aleatoria.

        Args:
            size: Cantidad de individuos en la poblacion.
            n_triangles: Cantidad de genes por individuo.
            gene_type: Tipo de gen a crear ("triangle" o "ellipse").

        Returns:
            Population con 'size' individuos aleatorios.
        """
        individuals = [Individual.random(n_triangles, gene_type) for _ in range(size)]
        return cls(individuals=individuals)

    @classmethod
    def smart_random(cls, size: int, n_triangles: int, gene_type: str,
                     target_image: np.ndarray) -> "Population":
        """Crea una poblacion con colores sampleados de la imagen target."""
        individuals = [
            Individual.smart_random(n_triangles, gene_type, target_image)
            for _ in range(size)
        ]
        return cls(individuals=individuals)

    @property
    def best(self) -> Individual:
        """Retorna el individuo con mayor fitness.

        Raises:
            ValueError: Si la poblacion esta vacia.
        """
        if not self.individuals:
            raise ValueError("La poblacion esta vacia, no hay mejor individuo.")
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def average_fitness(self) -> float:
        """Retorna el fitness promedio de la poblacion.

        Raises:
            ValueError: Si la poblacion esta vacia.
        """
        if not self.individuals:
            raise ValueError("La poblacion esta vacia, no se puede calcular promedio.")
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    @property
    def fitness_std(self) -> float:
        """Retorna la desviacion estandar del fitness de la poblacion.

        Raises:
            ValueError: Si la poblacion esta vacia.
        """
        if not self.individuals:
            raise ValueError("La poblacion esta vacia, no se puede calcular std.")
        fitnesses = np.array([ind.fitness for ind in self.individuals])
        return float(np.std(fitnesses))

    def evaluate_all(
        self,
        target: np.ndarray,
        renderer: Renderer,
        fitness_fn: FitnessFunction,
    ) -> None:
        """Calcula el fitness de todos los individuos de la poblacion.

        Args:
            target: Imagen objetivo, array numpy (H, W, 4), float32, [0, 1].
            renderer: Instancia de Renderer para convertir genes a imagen.
            fitness_fn: Instancia de FitnessFunction para calcular similitud.
        """
        for individual in self.individuals:
            individual.compute_fitness(target, renderer, fitness_fn)
