from genes.gene import Gene


class PolygonGene(Gene):
    """Clase base abstracta para genes con forma geometrica y color RGBA.

    Marcador de tipo sin campos propios. Tanto TriangleGene como EllipseGene
    definen sus propios campos (r, g, b, a) para evitar conflictos con
    herencia de dataclasses.
    """
    pass
