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
    """Genera genes aleatorios con colores sampleados de la imagen target.

    Las coordenadas y alpha son aleatorias como en random_genes().
    Los colores RGB se obtienen de pixeles aleatorios de la imagen target.

    Args:
        gene_type: "triangle" o "ellipse"
        n: cantidad de genes a generar
        target_image: imagen target, shape (H, W, 4), float32, [0, 1]

    Returns:
        Array (n, 11) con genes.
    """
    arr = random_genes(gene_type, n)

    h, w = target_image.shape[:2]
    ys = np.random.randint(0, h, size=n)
    xs = np.random.randint(0, w, size=n)
    sampled_rgb = target_image[ys, xs, :3]  # shape (n, 3), float32 en [0, 1]

    arr[:, 6:9] = np.round(sampled_rgb * 255.0)
    return arr


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
