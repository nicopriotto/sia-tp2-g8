"""
Metadata central para la representación numpy de genes.

Cada gen (triángulo o elipse) se almacena como un row de 11 floats.
Los índices 6-10 (r, g, b, a, active) son comunes a ambos tipos.

Triangle: [x1, y1, x2, y2, x3, y3, r, g, b, a, active]
Ellipse:  [cx, cy, rx, ry, theta, _pad, r, g, b, a, active]
"""

import math

import numpy as np

N_COLS = 11

TRIANGLE_COLS = ["x1", "y1", "x2", "y2", "x3", "y3", "r", "g", "b", "a", "active"]
ELLIPSE_COLS = ["cx", "cy", "rx", "ry", "theta", "_pad", "r", "g", "b", "a", "active"]

TRIANGLE_LOW = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
TRIANGLE_HIGH = np.array([1, 1, 1, 1, 1, 1, 255, 255, 255, 1, 1], dtype=np.float64)

ELLIPSE_LOW = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
ELLIPSE_HIGH = np.array([1, 1, 0.5, 0.5, math.pi, 0, 255, 255, 255, 1, 1], dtype=np.float64)

LAYOUTS = {
    "triangle": {"cols": TRIANGLE_COLS, "low": TRIANGLE_LOW, "high": TRIANGLE_HIGH},
    "ellipse": {"cols": ELLIPSE_COLS, "low": ELLIPSE_LOW, "high": ELLIPSE_HIGH},
}


def random_genes(gene_type: str, n: int) -> np.ndarray:
    """Genera array (n, 11) con genes aleatorios en rangos válidos."""
    layout = LAYOUTS[gene_type]
    low, high = layout["low"], layout["high"]
    arr = np.random.uniform(low, high, size=(n, N_COLS))
    # RGB deben ser enteros
    arr[:, 6:9] = np.round(arr[:, 6:9])
    # Active siempre 1.0 al crear
    arr[:, 10] = 1.0
    # Padding para ellipse
    if gene_type == "ellipse":
        arr[:, 5] = 0.0
    return arr



def smart_random_genes(gene_type: str, n: int, target_image: np.ndarray) -> np.ndarray:
    """Genera genes con inicialización inteligente multi-escala.

    Estrategia:
    1. Posiciones random (cobertura natural por overlap).
    2. Ordenados por tamaño: grandes primero (fondo), chicos último
       (detalle visible encima).
    3. Color híbrido: las formas grandes (top 60%) usan el color
       promedio de un parche de la imagen proporcional a su tamaño;
       las formas chicas (bottom 40%) sampleen varios puntos dentro
       de la forma y promedian, capturando mejor el detalle local.
    4. Alpha proporcional al tamaño: grandes semi-transparentes (0.30,
       se mezclan suavemente), chicos más opacos (0.90, definen detalle).

    Args:
        gene_type: "triangle" o "ellipse"
        n: cantidad de genes a generar
        target_image: imagen target, shape (H, W, 4), float32, [0, 1]

    Returns:
        Array (n, 11) con genes.
    """
    genes = random_genes(gene_type, n)

    h, w = target_image.shape[:2]
    max_dim = max(w, h)

    # --- Calcular tamaño de cada forma ---
    if gene_type == "triangle":
        x1, y1 = genes[:, 0], genes[:, 1]
        x2, y2 = genes[:, 2], genes[:, 3]
        x3, y3 = genes[:, 4], genes[:, 5]
        areas = np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    else:  # ellipse
        areas = math.pi * genes[:, 2] * genes[:, 3]  # π * rx * ry

    # --- Ordenar: grandes primero (fondo), chicos último (detalle) ---
    order = np.argsort(-areas)
    genes = genes[order]
    areas = areas[order]

    # --- Alpha proporcional al tamaño ---
    norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-8)
    genes[:, 9] = 0.90 - norm * 0.60  # chicos ~0.90, grandes ~0.30

    # --- Color híbrido: patch para grandes, multipoint para chicos ---
    split_idx = int(n * 0.60)

    for i in range(n):
        if i < split_idx:
            # Grandes: color promedio de un parche centrado en el centroide
            if gene_type == "triangle":
                cx = (genes[i, 0] + genes[i, 2] + genes[i, 4]) / 3.0
                cy = (genes[i, 1] + genes[i, 3] + genes[i, 5]) / 3.0
            else:
                cx, cy = genes[i, 0], genes[i, 1]
            r_px = max(1, int(math.sqrt(areas[i]) * max_dim * 0.5))
            cx_i = int(np.clip(cx * w, 0, w - 1))
            cy_i = int(np.clip(cy * h, 0, h - 1))
            y0 = max(0, cy_i - r_px)
            y1_ = min(h, cy_i + r_px + 1)
            x0 = max(0, cx_i - r_px)
            x1_ = min(w, cx_i + r_px + 1)
            patch = target_image[y0:y1_, x0:x1_, :3]
            genes[i, 6:9] = np.round(patch.mean(axis=(0, 1)) * 255.0)
        else:
            # Chicos: promedio de 10 puntos random dentro de la forma
            if gene_type == "triangle":
                ax, ay = genes[i, 0], genes[i, 1]
                bx, by = genes[i, 2], genes[i, 3]
                cvx, cvy = genes[i, 4], genes[i, 5]
                colors = np.empty((10, 3), dtype=np.float32)
                for s in range(10):
                    r1, r2 = np.random.random(), np.random.random()
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2
                    px = int(np.clip((ax + r1 * (bx - ax) + r2 * (cvx - ax)) * w, 0, w - 1))
                    py = int(np.clip((ay + r1 * (by - ay) + r2 * (cvy - ay)) * h, 0, h - 1))
                    colors[s] = target_image[py, px, :3]
            else:
                cx, cy = genes[i, 0], genes[i, 1]
                px = int(np.clip(cx * w, 0, w - 1))
                py = int(np.clip(cy * h, 0, h - 1))
                colors = target_image[py, px, :3].reshape(1, 3)
            genes[i, 6:9] = np.round(colors.mean(axis=0) * 255.0)

    return genes


def clamp(arr: np.ndarray, gene_type: str) -> np.ndarray:
    """Clampea valores al rango válido, in-place. Retorna arr."""
    np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
    layout = LAYOUTS[gene_type]
    np.clip(arr, layout["low"], layout["high"], out=arr)
    arr[:, 6:9] = np.round(arr[:, 6:9])
    arr[:, 10] = np.round(arr[:, 10]).clip(0, 1)
    return arr


def clamp_row(row: np.ndarray, gene_type: str) -> np.ndarray:
    """Clampea una sola fila (1D). Retorna row."""
    np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
    layout = LAYOUTS[gene_type]
    np.clip(row, layout["low"], layout["high"], out=row)
    row[6:9] = np.round(row[6:9])
    row[10] = round(min(1, max(0, row[10])))
    return row


def row_to_dict(row: np.ndarray, gene_type: str) -> dict:
    """Convierte una fila del array al formato dict para JSON."""
    cols = LAYOUTS[gene_type]["cols"]
    d = {}
    for i, name in enumerate(cols):
        if name == "_pad":
            continue
        val = row[i]
        if name in ("r", "g", "b"):
            d[name] = int(val)
        elif name == "active":
            d[name] = bool(val > 0.5)
        else:
            d[name] = float(val)
    if gene_type == "ellipse":
        d["type"] = "ellipse"
    return d


def is_degenerate(row: np.ndarray, epsilon: float = 1e-10) -> bool:
    """True si el triángulo codificado en row tiene área ~= 0."""
    x1, y1, x2, y2, x3, y3 = row[0], row[1], row[2], row[3], row[4], row[5]
    area = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0
    return area < epsilon
