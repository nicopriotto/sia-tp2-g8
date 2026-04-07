import numpy as np
import pytest

from genes import gene_layout
from core.individual import Individual
from core.population import Population


def _red_image(h=50, w=50) -> np.ndarray:
    """Imagen toda roja (1, 0, 0, 1) en float32."""
    img = np.zeros((h, w, 4), dtype=np.float32)
    img[:, :, 0] = 1.0  # R
    img[:, :, 3] = 1.0  # A
    return img


class TestSmartInit:
    def test_smart_colors_from_target(self):
        """Los colores se sampleen de la imagen target."""
        target = _red_image()
        genes = gene_layout.smart_random_genes("triangle", 20, target)
        # Toda la imagen es roja pura -> R=255, G=0, B=0
        np.testing.assert_array_equal(genes[:, 6], 255.0)
        np.testing.assert_array_equal(genes[:, 7], 0.0)
        np.testing.assert_array_equal(genes[:, 8], 0.0)

    def test_smart_coords_still_random(self):
        """Las coordenadas son aleatorias, no sampleadas."""
        target = _red_image()
        np.random.seed(1)
        genes1 = gene_layout.smart_random_genes("triangle", 20, target)
        np.random.seed(2)
        genes2 = gene_layout.smart_random_genes("triangle", 20, target)
        # Coordenadas deben ser diferentes entre batches
        assert not np.array_equal(genes1[:, :6], genes2[:, :6])

    def test_smart_alpha_still_random(self):
        """El alpha es aleatorio, no sampleado de la imagen."""
        target = _red_image()
        genes = gene_layout.smart_random_genes("triangle", 100, target)
        alphas = genes[:, 9]
        # Con 100 genes, el alpha deberia tener variacion
        assert alphas.std() > 0.1

    def test_smart_same_shape_as_random(self):
        """Misma shape y rangos que random_genes."""
        target = _red_image()
        smart = gene_layout.smart_random_genes("triangle", 10, target)
        regular = gene_layout.random_genes("triangle", 10)
        assert smart.shape == regular.shape
        assert smart.dtype == regular.dtype

    def test_smart_population_size(self):
        """Population.smart_random crea la cantidad correcta de individuos."""
        target = _red_image()
        pop = Population.smart_random(10, 5, "triangle", target)
        assert len(pop.individuals) == 10
        for ind in pop.individuals:
            assert ind.genes.shape == (5, 11)

    def test_smart_init_individual(self):
        """Individual.smart_random funciona correctamente."""
        target = _red_image()
        ind = Individual.smart_random(15, "triangle", target)
        assert ind.genes.shape == (15, 11)
        # Colores deben ser rojos
        np.testing.assert_array_equal(ind.genes[:, 6], 255.0)
