import numpy as np
import pytest

from core.individual import Individual
from mutation.gaussian_mutation import GaussianMutation


def _make_individual(n_genes: int = 20) -> Individual:
    """Individuo con genes en el medio del rango para evitar clamp."""
    ind = Individual.random(n_genes)
    # Poner coords en 0.5, RGB en 128, alpha en 0.5 para evitar saturacion
    ind.genes[:, :6] = 0.5
    ind.genes[:, 6:9] = 128.0
    ind.genes[:, 9] = 0.5
    return ind


class TestGaussianDecay:
    def test_decay_zero_no_change(self):
        """Con decay_b=0, sigma es igual en gen 0 y gen 999."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=0.1, decay_b=0.0)
        np.random.seed(42)

        diffs_early = []
        for _ in range(50):
            ind = _make_individual()
            result = mut.mutate(ind, generation=0, max_generations=1000)
            diffs_early.append(np.abs(result.genes[:, :6] - ind.genes[:, :6]).mean())

        np.random.seed(42)
        diffs_late = []
        for _ in range(50):
            ind = _make_individual()
            result = mut.mutate(ind, generation=999, max_generations=1000)
            diffs_late.append(np.abs(result.genes[:, :6] - ind.genes[:, :6]).mean())

        # Sin decay, las perturbaciones deben ser iguales (misma seed)
        assert np.mean(diffs_early) == pytest.approx(np.mean(diffs_late), rel=0.01)

    def test_decay_reduces_perturbation(self):
        """Con decay_b=2, la perturbacion en gen 90 es mucho menor que en gen 0."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=0.5, decay_b=2.0)

        diffs_early = []
        for _ in range(100):
            ind = _make_individual()
            result = mut.mutate(ind, generation=0, max_generations=100)
            diffs_early.append(np.abs(result.genes[:, :6] - ind.genes[:, :6]).mean())

        diffs_late = []
        for _ in range(100):
            ind = _make_individual()
            result = mut.mutate(ind, generation=90, max_generations=100)
            diffs_late.append(np.abs(result.genes[:, :6] - ind.genes[:, :6]).mean())

        mean_early = np.mean(diffs_early)
        mean_late = np.mean(diffs_late)
        # La perturbacion tardia debe ser < 50% de la temprana
        assert mean_late < mean_early * 0.5

    def test_decay_last_generation_near_zero(self):
        """En generation=max_generations con decay, el individuo casi no cambia."""
        mut = GaussianMutation(mutation_rate=1.0, sigma=0.5, decay_b=1.0)
        ind = _make_individual()
        result = mut.mutate(ind, generation=100, max_generations=100)

        # sigma_eff = 0.5 * (1 - 1.0)^1 = 0 -> sin perturbacion
        np.testing.assert_array_almost_equal(result.genes, ind.genes)

    def test_decay_does_not_affect_mutation_rate(self):
        """El decay solo afecta magnitud, no la probabilidad de seleccion."""
        mut = GaussianMutation(mutation_rate=0.5, sigma=0.5, decay_b=2.0)

        # Contar cuantos genes mutan en gen temprana vs tardia
        count_early = 0
        count_late = 0
        trials = 200
        n_genes = 20

        for _ in range(trials):
            ind = _make_individual(n_genes)
            result = mut.mutate(ind, generation=0, max_generations=100)
            count_early += np.any(result.genes[:, :6] != ind.genes[:, :6], axis=1).sum()

        for _ in range(trials):
            ind = _make_individual(n_genes)
            result = mut.mutate(ind, generation=90, max_generations=100)
            # Los cambios son minusculos pero no exactamente cero (sigma > 0)
            count_late += np.any(result.genes[:, :6] != ind.genes[:, :6], axis=1).sum()

        rate_early = count_early / (trials * n_genes)
        rate_late = count_late / (trials * n_genes)
        # Ambas deben estar cerca de 0.5
        assert 0.35 < rate_early < 0.65
        assert 0.35 < rate_late < 0.65
