import numpy as np
from PIL import Image, ImageDraw
from render.renderer import Renderer
from genes.triangle_gene import TriangleGene


class CPURenderer(Renderer):
    """Renderer CPU que usa Pillow para dibujar triangulos con alpha compositing."""

    def render(self, genes: list[TriangleGene], width: int, height: int) -> np.ndarray:
        """
        Renderiza una lista de TriangleGene sobre un canvas blanco.

        Algoritmo:
        1. Crear canvas RGBA blanco (255, 255, 255, 255) de tamano (width, height).
        2. Por cada gen en orden:
           a. Convertir coords normalizadas a pixeles.
           b. Crear capa temporal RGBA transparente (0, 0, 0, 0).
           c. Dibujar el triangulo en la capa temporal con su color y alpha.
           d. Componer la capa sobre el canvas con Image.alpha_composite.
        3. Convertir el canvas a numpy float32 dividiendo por 255.0.
        """
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

        for gene in genes:
            points = [
                (int(gene.x1 * width), int(gene.y1 * height)),
                (int(gene.x2 * width), int(gene.y2 * height)),
                (int(gene.x3 * width), int(gene.y3 * height)),
            ]

            color = (gene.r, gene.g, gene.b, int(gene.a * 255))

            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            draw.polygon(points, fill=color)

            canvas = Image.alpha_composite(canvas, layer)

        result = np.array(canvas, dtype=np.float32) / 255.0
        return result
