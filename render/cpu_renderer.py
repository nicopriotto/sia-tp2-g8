import math
import threading

import numpy as np
from PIL import Image, ImageDraw
from render.renderer import Renderer


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

    def render(self, genes: np.ndarray, width: int, height: int, gene_type: str = "triangle") -> np.ndarray:
        """
        Renderiza un array de genes sobre un canvas blanco.

        Canvas es numpy float32 desde el inicio. Compositing en numpy in-place.
        """
        canvas = np.ones((height, width, 4), dtype=np.float32)

        if len(genes) == 0:
            return canvas

        layer, draw = self._get_layer(width, height)

        for row in genes:
            if row[10] < 0.5:  # inactive
                continue

            if gene_type == "ellipse":
                self._draw_ellipse_row(canvas, row, width, height, layer, draw)
            else:
                self._draw_triangle_row(canvas, row, width, height, layer, draw)

        return canvas

    def _composite(self, canvas: np.ndarray, layer_pil: Image.Image):
        """Alpha composite layer_pil onto canvas (numpy) in-place."""
        layer_np = np.array(layer_pil, dtype=np.float32) / 255.0
        alpha = layer_np[:, :, 3:4]
        canvas[:, :, :3] *= (1.0 - alpha)
        canvas[:, :, :3] += layer_np[:, :, :3] * alpha

    def _draw_triangle_row(self, canvas: np.ndarray, row: np.ndarray,
                           width: int, height: int,
                           layer: Image.Image, draw: ImageDraw.ImageDraw):
        """Dibuja un triangulo sobre el canvas con numpy compositing."""
        draw.rectangle((0, 0, width - 1, height - 1), fill=(0, 0, 0, 0))
        points = [
            (int(row[0] * width), int(row[1] * height)),
            (int(row[2] * width), int(row[3] * height)),
            (int(row[4] * width), int(row[5] * height)),
        ]
        color = (int(row[6]), int(row[7]), int(row[8]), int(row[9] * 255))
        draw.polygon(points, fill=color)
        self._composite(canvas, layer)

    def _draw_ellipse_row(self, canvas: np.ndarray, row: np.ndarray,
                          width: int, height: int,
                          layer: Image.Image, draw: ImageDraw.ImageDraw):
        """Dibuja una elipse rotada sobre el canvas con numpy compositing."""
        draw.rectangle((0, 0, width - 1, height - 1), fill=(0, 0, 0, 0))

        cx_px = int(row[0] * width)
        cy_px = int(row[1] * height)
        px_rx = int(row[2] * width)
        px_ry = int(row[3] * height)
        theta_deg = math.degrees(row[4])
        color = (int(row[6]), int(row[7]), int(row[8]), int(row[9] * 255))

        bbox = [
            cx_px - px_rx, cy_px - px_ry,
            cx_px + px_rx, cy_px + px_ry,
        ]
        draw.ellipse(bbox, fill=color)

        if theta_deg != 0:
            rotated = layer.rotate(-theta_deg, center=(cx_px, cy_px), resample=Image.BILINEAR)
            self._composite(canvas, rotated)
        else:
            self._composite(canvas, layer)
