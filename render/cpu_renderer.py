import math

import numpy as np
from PIL import Image, ImageDraw
from render.renderer import Renderer
from genes.triangle_gene import TriangleGene
from genes.ellipse_gene import EllipseGene


class CPURenderer(Renderer):
    """Renderer CPU que usa Pillow para dibujar triangulos y elipses con alpha compositing."""

    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """
        Renderiza una lista de genes (TriangleGene o EllipseGene) sobre un canvas blanco.

        Algoritmo:
        1. Crear canvas RGBA blanco (255, 255, 255, 255) de tamano (width, height).
        2. Por cada gen en orden, detectar el tipo y dibujar con el metodo correspondiente.
        3. Convertir el canvas a numpy float32 dividiendo por 255.0.
        """
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

        for gene in genes:
            # Saltar genes inactivos
            if hasattr(gene, 'active') and not gene.active:
                continue

            if isinstance(gene, EllipseGene):
                canvas = self._draw_ellipse(canvas, gene, width, height)
            elif isinstance(gene, TriangleGene):
                canvas = self._draw_triangle(canvas, gene, width, height)

        result = np.array(canvas, dtype=np.float32) / 255.0
        return result

    def _draw_triangle(self, canvas: Image.Image, gene: TriangleGene, width: int, height: int) -> Image.Image:
        """Dibuja un triangulo sobre el canvas con alpha compositing."""
        points = [
            (int(gene.x1 * width), int(gene.y1 * height)),
            (int(gene.x2 * width), int(gene.y2 * height)),
            (int(gene.x3 * width), int(gene.y3 * height)),
        ]
        color = (gene.r, gene.g, gene.b, int(gene.a * 255))
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        draw.polygon(points, fill=color)
        return Image.alpha_composite(canvas, layer)

    def _draw_ellipse(self, canvas: Image.Image, gene: EllipseGene, width: int, height: int) -> Image.Image:
        """Dibuja una elipse rotada sobre el canvas con alpha compositing."""
        px_rx = int(gene.rx * width)
        px_ry = int(gene.ry * height)
        color = (gene.r, gene.g, gene.b, int(gene.a * 255))

        cx_px = int(gene.cx * width)
        cy_px = int(gene.cy * height)
        bbox = [
            cx_px - px_rx, cy_px - px_ry,
            cx_px + px_rx, cy_px + px_ry,
        ]

        # Dibujar elipse sin rotacion en una capa temporal
        temp = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp)
        draw.ellipse(bbox, fill=color)

        # Rotar alrededor del centro de la elipse
        theta_deg = math.degrees(gene.theta)
        rotated = temp.rotate(-theta_deg, center=(cx_px, cy_px), resample=Image.BILINEAR)

        return Image.alpha_composite(canvas, rotated)
