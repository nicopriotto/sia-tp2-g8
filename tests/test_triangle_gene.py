from unittest.mock import patch
from genes.triangle_gene import TriangleGene


def test_random_values_in_range():
    for _ in range(100):
        g = TriangleGene.random()
        for attr in ["x1", "y1", "x2", "y2", "x3", "y3", "a"]:
            assert 0.0 <= getattr(g, attr) <= 1.0
        for attr in ["r", "g", "b"]:
            assert 0 <= getattr(g, attr) <= 255


def test_copy_independence():
    original = TriangleGene.random()
    copied = original.copy()
    original.r = 0
    original.x1 = 0.999
    assert copied.r != 0 or copied.x1 != 0.999


def test_mutate_replace_valid():
    original = TriangleGene(
        x1=0.5, y1=0.5, x2=0.5, y2=0.5, x3=0.5, y3=0.5,
        r=128, g=128, b=128, a=0.5,
    )
    new = original.mutate_replace()
    assert new is not original
    for attr in ["x1", "y1", "x2", "y2", "x3", "y3", "a"]:
        assert 0.0 <= getattr(new, attr) <= 1.0
    for attr in ["r", "g", "b"]:
        assert 0 <= getattr(new, attr) <= 255


def test_mutate_delta_zero_strength():
    original = TriangleGene(
        x1=0.5, y1=0.3, x2=0.7, y2=0.1, x3=0.2, y3=0.9,
        r=100, g=150, b=200, a=0.6,
    )
    result = original.mutate_delta(0.0)
    assert result.x1 == original.x1
    assert result.y1 == original.y1
    assert result.x2 == original.x2
    assert result.y2 == original.y2
    assert result.x3 == original.x3
    assert result.y3 == original.y3
    assert result.r == original.r
    assert result.g == original.g
    assert result.b == original.b
    assert result.a == original.a


def test_mutate_delta_clamp():
    gene = TriangleGene(
        x1=0.99, y1=0.01, x2=0.5, y2=0.5, x3=0.5, y3=0.5,
        r=250, g=5, b=128, a=0.95,
    )
    for _ in range(100):
        result = gene.mutate_delta(0.5)
        for attr in ["x1", "y1", "x2", "y2", "x3", "y3", "a"]:
            assert 0.0 <= getattr(result, attr) <= 1.0
        for attr in ["r", "g", "b"]:
            assert 0 <= getattr(result, attr) <= 255


def test_is_degenerate_collinear():
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=0.5, y2=0.5, x3=1.0, y3=1.0,
        r=128, g=128, b=128, a=0.5,
    )
    assert gene._is_degenerate() is True


def test_is_degenerate_two_equal_points():
    gene = TriangleGene(
        x1=0.3, y1=0.3, x2=0.3, y2=0.3, x3=0.7, y3=0.9,
        r=128, g=128, b=128, a=0.5,
    )
    assert gene._is_degenerate() is True


def test_mutate_delta_never_returns_degenerate():
    gene = TriangleGene(
        x1=0.2, y1=0.1, x2=0.8, y2=0.3, x3=0.5, y3=0.9,
        r=100, g=150, b=200, a=0.5,
    )
    for _ in range(1000):
        result = gene.mutate_delta(0.01)
        assert result._is_degenerate() is False


def test_mutate_delta_degenerate_does_not_use_random():
    gene = TriangleGene(
        x1=0.0, y1=0.0, x2=0.5, y2=0.5, x3=1.0, y3=1.0,
        r=128, g=128, b=128, a=0.5,
    )
    result = gene.mutate_delta(0.01)
    if not result._is_degenerate():
        # El resultado debe ser cercano al original, no un gen aleatorio lejano
        assert abs(result.r - gene.r) < 128
        assert abs(result.a - gene.a) < 0.5


def test_mutate_delta_degenerate_falls_back_to_original():
    gene = TriangleGene(
        x1=0.2, y1=0.1, x2=0.8, y2=0.3, x3=0.5, y3=0.9,
        r=100, g=150, b=200, a=0.5,
    )
    # Mockear _apply_delta para que siempre retorne un gen degenerado
    degenerate = TriangleGene(
        x1=0.0, y1=0.0, x2=0.5, y2=0.5, x3=1.0, y3=1.0,
        r=100, g=150, b=200, a=0.5,
    )
    with patch.object(TriangleGene, "_apply_delta", return_value=degenerate):
        result = gene.mutate_delta(0.01)
    # Debe ser una copia del original
    assert result.x1 == gene.x1
    assert result.y1 == gene.y1
    assert result.x2 == gene.x2
    assert result.r == gene.r
    assert result is not gene


def test_to_dict_keys():
    gene = TriangleGene.random()
    d = gene.to_dict()
    expected_keys = {"x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a"}
    assert set(d.keys()) == expected_keys
