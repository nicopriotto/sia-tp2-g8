import numpy as np
import pytest

from core.individual import Individual
from mutation.gaussian_mutation import GaussianMutation
from genes import gene_layout


def _make_individual(n_genes: int = 20, gene_type: str = "triangle") -> Individual:
    """Individuo con valores en el medio del rango para evitar clamp."""
    ind = Individual.random(n_genes, gene_type)
    ind.genes[:, :6] = 0.5
    ind.genes[:, 6:9] = 128.0
    ind.genes[:, 9] = 0.5
    ind.genes[:, 10] = 1.0
    return ind


class TestGaussianPerFloat:
    def test_per_float_partial_mutation(self):
        """Algunos floats cambian y otros no dentro del mismo triangulo."""
        mut = GaussianMutation(mutation_rate=0.5, sigma=0.1)
        ind = _make_individual(10)

        # Repetir hasta encontrar un triangulo parcialmente mutado
        found_partial = False
        for _ in range(50):
            result = mut.mutate(ind, generation=0, max_generations=100)
            for row in range(10):
                # Comparar columnas 0-9 (excluir active)
                diffs = result.genes[row, :10] != ind.genes[row, :10]
                if diffs.any() and not diffs.all():
                    found_partial = True
                    break
            if found_partial:
                break
        assert found_partial, "Nunca se encontro un triangulo parcialmente mutado"

    def test_active_never_mutated(self):
        """La columna 10 (active) nunca cambia."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=0.5)
        ind = _make_individual()
        original_active = ind.genes[:, 10].copy()

        for _ in range(20):
            result = mut.mutate(ind, generation=0, max_generations=100)
            np.testing.assert_array_equal(result.genes[:, 10], original_active)

    def test_ellipse_padding_never_mutated(self):
        """La columna 5 (padding ellipse) nunca cambia."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=0.5)
        ind = _make_individual(gene_type="ellipse")
        ind.genes[:, 5] = 0.0

        for _ in range(20):
            result = mut.mutate(ind, generation=0, max_generations=100)
            np.testing.assert_array_equal(result.genes[:, 5], 0.0)

    def test_mutation_rate_zero(self):
        """Con mutation_rate=0, genes identicos."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.5)
        ind = _make_individual()
        result = mut.mutate(ind, generation=0, max_generations=100)
        np.testing.assert_array_equal(result.genes, ind.genes)

    def test_mutation_rate_one_all_change(self):
        """Con mutation_rate=1 y sigma alto, todas las columnas 0-9 cambian."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=1.0, sigma_color=1.0)
        ind = _make_individual()

        result = mut.mutate(ind, generation=0, max_generations=100)
        # Verificar que cada columna 0-9 tiene al menos algun cambio
        for col in range(10):
            diffs = result.genes[:, col] != ind.genes[:, col]
            assert diffs.any(), f"Columna {col} no cambio con mutation_rate=1.0"

    def test_scales_correct(self):
        """Las escalas de ruido son correctas por columna."""
        sigma = 0.1
        sigma_color = 0.05
        mut = GaussianMutation(mutation_rate=1.0, sigma=sigma, sigma_color=sigma_color)

        coord_diffs = []
        rgb_diffs = []
        alpha_diffs = []

        for _ in range(500):
            ind = _make_individual()
            result = mut.mutate(ind, generation=0, max_generations=100)
            coord_diffs.append((result.genes[:, :6] - ind.genes[:, :6]).ravel())
            rgb_diffs.append((result.genes[:, 6:9] - ind.genes[:, 6:9]).ravel())
            alpha_diffs.append((result.genes[:, 9] - ind.genes[:, 9]).ravel())

        coord_std = np.std(np.concatenate(coord_diffs))
        rgb_std = np.std(np.concatenate(rgb_diffs))
        alpha_std = np.std(np.concatenate(alpha_diffs))

        # Tolerancia del 25% (clamp afecta la distribucion)
        assert abs(coord_std - sigma) / sigma < 0.25
        assert abs(rgb_std - sigma_color * 255) / (sigma_color * 255) < 0.25
        assert abs(alpha_std - sigma_color) / sigma_color < 0.25

    def test_clamp_respected(self):
        """Con sigma exagerado, los valores quedan en rango valido."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=10.0, sigma_color=10.0)
        ind = _make_individual()
        result = mut.mutate(ind, generation=0, max_generations=100)

        layout = gene_layout.LAYOUTS["triangle"]
        for col in range(gene_layout.N_COLS):
            assert np.all(result.genes[:, col] >= layout["low"][col])
            assert np.all(result.genes[:, col] <= layout["high"][col])
