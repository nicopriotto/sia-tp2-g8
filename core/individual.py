from dataclasses import dataclass
import numpy as np
from genes.gene import Gene
from genes.triangle_gene import TriangleGene
from genes.ellipse_gene import EllipseGene
from render.renderer import Renderer
from fitness.fitness_function import FitnessFunction

# Mapa de tipo de gen a clase concreta
_GENE_TYPES: dict[str, type[Gene]] = {
    "triangle": TriangleGene,
    "ellipse": EllipseGene,
}


@dataclass
class Individual:
    """Representa un individuo del algoritmo genetico.

    Un individuo es una lista ordenada de genes (triangulos o elipses) y su fitness.
    El fitness se inicializa en 0.0 y se actualiza con compute_fitness().
    """
    genes: list[Gene]
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
        gene_cls = _GENE_TYPES.get(gene_type, TriangleGene)
        genes = [gene_cls.random() for _ in range(n_triangles)]
        return cls(genes=genes)

    def copy(self) -> "Individual":
        """Retorna una copia profunda del individuo.

        Cada gen se copia independientemente para evitar referencias compartidas.
        El fitness se copia tal cual.
        """
        copied_genes = [gene.copy() for gene in self.genes]
        return Individual(genes=copied_genes, fitness=self.fitness, fitness_valid=self.fitness_valid)

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
        height, width = target.shape[0], target.shape[1]
        generated = renderer.render(self.genes, width, height)
        self.fitness = fitness_fn.compute(generated, target)
        self.fitness_valid = True

    def to_dict(self) -> dict:
        """Serializa el individuo a un diccionario.

        Returns:
            dict con claves 'genes' (lista de dicts) y 'fitness' (float).
        """
        return {
            "genes": [gene.to_dict() for gene in self.genes],
            "fitness": self.fitness,
        }
