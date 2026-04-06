import math
import random

import numpy as np

from genes.ellipse_gene import EllipseGene
from genes.polygon_gene import PolygonGene
from genes.triangle_gene import TriangleGene
from render.cpu_renderer import CPURenderer


def test_ellipse_gene_values_in_range():
    """Genera 100 EllipseGene.random() y verifica rangos."""
    for _ in range(100):
        g = EllipseGene.random()
        assert 0.0 <= g.cx <= 1.0
        assert 0.0 <= g.cy <= 1.0
        assert 0.0 <= g.rx <= 0.5
        assert 0.0 <= g.ry <= 0.5
        assert 0.0 <= g.theta <= math.pi
        assert 0 <= g.r <= 255
        assert 0 <= g.g <= 255
        assert 0 <= g.b <= 255
        assert 0.0 <= g.a <= 1.0


def test_ellipse_copy_independence():
    """Crea un gen, lo copia, modifica el original. La copia no se afecta."""
    original = EllipseGene.random()
    copied = original.copy()
    original.cx = 0.999
    original.r = 0
    assert copied.cx != 0.999 or copied.r != 0


def test_ellipse_mutate_delta_zero():
    """Aplica mutate_delta(0.0) con seed fijo. El resultado debe ser identico."""
    gene = EllipseGene(
        cx=0.5, cy=0.3, rx=0.2, ry=0.1,
        theta=1.0, r=100, g=150, b=200, a=0.6,
    )
    # Fijar seed para evitar flip de active (prob 5%)
    random.seed(42)
    result = gene.mutate_delta(0.0)
    assert result.cx == gene.cx
    assert result.cy == gene.cy
    assert result.rx == gene.rx
    assert result.ry == gene.ry
    assert result.theta == gene.theta
    assert result.r == gene.r
    assert result.g == gene.g
    assert result.b == gene.b
    assert result.a == gene.a


def test_ellipse_mutate_delta_clamp():
    """Crea un gen con valores extremos y aplica delta. Verifica que sigue en rango."""
    gene = EllipseGene(
        cx=0.99, cy=0.01, rx=0.49, ry=0.01,
        theta=0.01, r=250, g=5, b=128, a=0.95,
    )
    for _ in range(100):
        result = gene.mutate_delta(0.5)
        assert 0.0 <= result.cx <= 1.0
        assert 0.0 <= result.cy <= 1.0
        assert 0.0 <= result.rx <= 0.5
        assert 0.0 <= result.ry <= 0.5
        assert 0.0 <= result.theta <= math.pi
        assert 0 <= result.r <= 255
        assert 0 <= result.g <= 255
        assert 0 <= result.b <= 255
        assert 0.0 <= result.a <= 1.0


def test_ellipse_mutate_replace_valid():
    """Llama mutate_replace() y verifica rangos validos y diferencia."""
    original = EllipseGene(
        cx=0.5, cy=0.5, rx=0.25, ry=0.25,
        theta=1.0, r=128, g=128, b=128, a=0.5,
    )
    new = original.mutate_replace()
    assert new is not original
    assert 0.0 <= new.cx <= 1.0
    assert 0.0 <= new.cy <= 1.0
    assert 0.0 <= new.rx <= 0.5
    assert 0.0 <= new.ry <= 0.5
    assert 0.0 <= new.theta <= math.pi
    assert 0 <= new.r <= 255
    assert 0 <= new.g <= 255
    assert 0 <= new.b <= 255
    assert 0.0 <= new.a <= 1.0


def test_renderer_with_ellipse_shape():
    """Renderiza con EllipseGene y verifica shape del resultado."""
    renderer = CPURenderer()
    gene = EllipseGene(
        cx=0.5, cy=0.5, rx=0.2, ry=0.2,
        theta=0.0, r=255, g=0, b=0, a=1.0,
    )
    result = renderer.render([gene], 100, 80)
    assert result.shape == (80, 100, 4)


def test_renderer_with_ellipse_range():
    """Renderiza varios EllipseGene aleatorios. Todos los valores en [0, 1]."""
    renderer = CPURenderer()
    genes = [EllipseGene.random() for _ in range(10)]
    result = renderer.render(genes, 64, 64)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_ellipse_opaque_modifies_canvas():
    """Crea una elipse roja opaca grande centrada. Verifica que pixeles rojos existen."""
    renderer = CPURenderer()
    gene = EllipseGene(
        cx=0.5, cy=0.5, rx=0.4, ry=0.4,
        theta=0.0, r=255, g=0, b=0, a=1.0,
    )
    result = renderer.render([gene], 100, 100)
    # Al menos algunos pixeles deben tener canal R cercano a 1.0 y G, B cercanos a 0.0
    red_mask = (result[:, :, 0] > 0.9) & (result[:, :, 1] < 0.1) & (result[:, :, 2] < 0.1)
    assert red_mask.sum() > 0, "La elipse roja opaca no se dibujo en el canvas"


def test_triangle_gene_unaffected():
    """Verifica que TriangleGene hereda de PolygonGene y los tests existentes siguen validos."""
    # Verificar herencia
    assert issubclass(TriangleGene, PolygonGene)
    assert isinstance(TriangleGene.random(), PolygonGene)

    # Verificar que random() sigue generando valores en rango
    for _ in range(100):
        g = TriangleGene.random()
        for attr in ["x1", "y1", "x2", "y2", "x3", "y3", "a"]:
            assert 0.0 <= getattr(g, attr) <= 1.0
        for attr in ["r", "g", "b"]:
            assert 0 <= getattr(g, attr) <= 255

    # Verificar copy
    original = TriangleGene.random()
    copied = original.copy()
    original.r = 0
    original.x1 = 0.999
    assert copied.r != 0 or copied.x1 != 0.999

    # Verificar to_dict
    gene = TriangleGene.random()
    d = gene.to_dict()
    expected_keys = {"x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a", "active"}
    assert set(d.keys()) == expected_keys


def test_ellipse_to_dict():
    """Verifica serializacion de EllipseGene."""
    gene = EllipseGene(
        cx=0.5, cy=0.5, rx=0.2, ry=0.3,
        theta=1.0, r=100, g=150, b=200, a=0.7,
    )
    d = gene.to_dict()
    assert d["type"] == "ellipse"
    assert d["cx"] == 0.5
    assert d["cy"] == 0.5
    assert d["rx"] == 0.2
    assert d["ry"] == 0.3
    assert d["theta"] == 1.0
    assert d["r"] == 100
    assert d["g"] == 150
    assert d["b"] == 200
    assert d["a"] == 0.7
    assert d["active"] is True


def test_ellipse_inherits_polygon_gene():
    """Verifica que EllipseGene hereda de PolygonGene."""
    assert issubclass(EllipseGene, PolygonGene)
    assert isinstance(EllipseGene.random(), PolygonGene)
