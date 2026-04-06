import math
import threading

import numpy as np
from PIL import Image, ImageDraw
from render.renderer import Renderer
from genes.triangle_gene import TriangleGene
from genes.ellipse_gene import EllipseGene


class CPURenderer(Renderer):
    """Renderer CPU que usa Pillow para dibujar y numpy para alpha compositing."""

    def __init__(self):
        self._local = threading.local()

    def _get_layer(self, width: int, height: int):
        """Retorna (layer, draw) thread-local, creando si es necesario."""
        local = self._local
        if not hasattr(local, 'layer') or local.layer.size != (width, height):
            local.layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            local.draw = ImageDraw.Draw(local.layer)
        return local.layer, local.draw

    def render(self, genes: list, width: int, height: int) -> np.ndarray:
        """
        Renderiza una lista de genes (TriangleGene o EllipseGene) sobre un canvas blanco.

        Canvas es numpy float32 desde el inicio. Compositing en numpy in-place.
        """
        canvas = np.ones((height, width, 4), dtype=np.float32)
        layer, draw = self._get_layer(width, height)

        for gene in genes:
            if hasattr(gene, 'active') and not gene.active:
                continue

            if isinstance(gene, EllipseGene):
                self._draw_ellipse_onto(canvas, gene, width, height, layer, draw)
            elif isinstance(gene, TriangleGene):
                self._draw_triangle_onto(canvas, gene, width, height, layer, draw)

        return canvas

    def _composite(self, canvas: np.ndarray, layer_pil: Image.Image):
        """Alpha composite layer_pil onto canvas (numpy) in-place."""
        layer_np = np.array(layer_pil, dtype=np.float32) / 255.0
        alpha = layer_np[:, :, 3:4]
        canvas[:, :, :3] *= (1.0 - alpha)
        canvas[:, :, :3] += layer_np[:, :, :3] * alpha

    def _draw_triangle_onto(self, canvas: np.ndarray, gene: TriangleGene,
                            width: int, height: int,
                            layer: Image.Image, draw: ImageDraw.ImageDraw):
        """Dibuja un triangulo sobre el canvas con numpy compositing."""
        draw.rectangle((0, 0, width - 1, height - 1), fill=(0, 0, 0, 0))
        points = [
            (int(gene.x1 * width), int(gene.y1 * height)),
            (int(gene.x2 * width), int(gene.y2 * height)),
            (int(gene.x3 * width), int(gene.y3 * height)),
        ]
        color = (gene.r, gene.g, gene.b, int(gene.a * 255))
        draw.polygon(points, fill=color)
        self._composite(canvas, layer)

    def _draw_ellipse_onto(self, canvas: np.ndarray, gene: EllipseGene,
                           width: int, height: int,
                           layer: Image.Image, draw: ImageDraw.ImageDraw):
        """Dibuja una elipse rotada sobre el canvas con numpy compositing."""
        draw.rectangle((0, 0, width - 1, height - 1), fill=(0, 0, 0, 0))

        px_rx = int(gene.rx * width)
        px_ry = int(gene.ry * height)
        color = (gene.r, gene.g, gene.b, int(gene.a * 255))

        cx_px = int(gene.cx * width)
        cy_px = int(gene.cy * height)
        bbox = [
            cx_px - px_rx, cy_px - px_ry,
            cx_px + px_rx, cy_px + px_ry,
        ]
        draw.ellipse(bbox, fill=color)

        theta_deg = math.degrees(gene.theta)
        if theta_deg != 0:
            rotated = layer.rotate(-theta_deg, center=(cx_px, cy_px), resample=Image.BILINEAR)
            self._composite(canvas, rotated)
        else:
            self._composite(canvas, layer)
