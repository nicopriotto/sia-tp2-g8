import numpy as np
import pytest

from core.individual import Individual
from mutation.gaussian_mutation import GaussianMutation


def _make_individual(n_genes: int = 10) -> Individual:
    """Individuo con genes distintos entre si para detectar swaps."""
    ind = Individual.random(n_genes)
    # Hacer cada fila unica usando el indice como x1
    for i in range(n_genes):
        ind.genes[i, 0] = i / n_genes
    return ind


class TestSwapZIndex:
    def test_swap_rate_zero_no_change(self):
        """Sin perturbacion ni swap, genes identicos."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.0, swap_rate=0.0)
        ind = _make_individual()
        result = mut.mutate(ind, generation=0, max_generations=100)
        np.testing.assert_array_equal(result.genes, ind.genes)

    def test_swap_preserves_genes(self):
        """El swap solo reordena filas, no cambia valores."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.0, swap_rate=1.0)
        ind = _make_individual()
        result = mut.mutate(ind, generation=0, max_generations=100)

        # Ordenar ambos por axis=0 para comparar contenido sin importar orden
        orig_sorted = np.sort(ind.genes, axis=0)
        result_sorted = np.sort(result.genes, axis=0)
        np.testing.assert_array_equal(result_sorted, orig_sorted)

    def test_swap_changes_order(self):
        """Con swap_rate=1, al menos dos filas cambian de posicion."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.0, swap_rate=1.0)
        ind = _make_individual()

        changed = False
        for _ in range(20):
            result = mut.mutate(ind, generation=0, max_generations=100)
            if not np.array_equal(result.genes, ind.genes):
                changed = True
                break
        assert changed, "El swap nunca cambio el orden en 20 intentos"

    def test_swap_only_two_rows(self):
        """Exactamente 2 filas cambian de posicion, el resto igual."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.0, swap_rate=1.0)
        ind = _make_individual(20)
        result = mut.mutate(ind, generation=0, max_generations=100)

        # Contar filas que cambiaron
        changed_rows = 0
        for i in range(ind.genes.shape[0]):
            if not np.array_equal(result.genes[i], ind.genes[i]):
                changed_rows += 1

        assert changed_rows == 2 or changed_rows == 0  # 0 si i==j por azar (replace=False lo evita)
        if changed_rows == 2:
            # Verificar que son un swap (las filas intercambiaron)
            diff_indices = [i for i in range(ind.genes.shape[0])
                           if not np.array_equal(result.genes[i], ind.genes[i])]
            i, j = diff_indices
            np.testing.assert_array_equal(result.genes[i], ind.genes[j])
            np.testing.assert_array_equal(result.genes[j], ind.genes[i])

    def test_swap_invalidates_fitness(self):
        """Despues de un swap, fitness_valid es False."""
        mut = GaussianMutation(mutation_rate=0.0, sigma=0.0, swap_rate=1.0)
        ind = _make_individual()
        ind.fitness = 0.9
        ind.fitness_valid = True
        result = mut.mutate(ind, generation=0, max_generations=100)
        assert result.fitness_valid is False
