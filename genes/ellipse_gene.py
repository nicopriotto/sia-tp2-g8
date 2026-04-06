import math
import random
from dataclasses import dataclass

import numpy as np

from genes.polygon_gene import PolygonGene


@dataclass
class EllipseGene(PolygonGene):
    """Gen que codifica una elipse semi-transparente rotada."""
    cx: float    # centro x, [0, 1]
    cy: float    # centro y, [0, 1]
    rx: float    # semieje x, [0, 0.5]
    ry: float    # semieje y, [0, 0.5]
    theta: float  # rotacion en radianes, [0, pi]
    r: int       # [0, 255]
    g: int       # [0, 255]
    b: int       # [0, 255]
    a: float     # [0, 1]
    active: bool = True  # gen activo o inactivo (elipse se dibuja o no)

    @classmethod
    def random(cls) -> "EllipseGene":
        """Crea un EllipseGene con valores aleatorios dentro de los rangos validos."""
        return cls(
            cx=random.random(),
            cy=random.random(),
            rx=random.uniform(0.0, 0.5),
            ry=random.uniform(0.0, 0.5),
            theta=random.uniform(0.0, math.pi),
            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            a=random.random(),
        )

    def copy(self) -> "EllipseGene":
        """Retorna una copia independiente del gen."""
        return EllipseGene(
            cx=self.cx, cy=self.cy,
            rx=self.rx, ry=self.ry,
            theta=self.theta,
            r=self.r, g=self.g, b=self.b,
            a=self.a, active=self.active,
        )

    def mutate_replace(self) -> "EllipseGene":
        """Genera un gen completamente nuevo con valores aleatorios."""
        return EllipseGene.random()

    def mutate_delta(self, strength: float) -> "EllipseGene":
        """Perturba atributos con delta proporcional a strength, clampea a rangos."""
        def df(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val + random.uniform(-strength, strength)))

        def di(val: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, round(val + random.uniform(-strength * 255, strength * 255))))

        result = EllipseGene(
            cx=df(self.cx, 0.0, 1.0),
            cy=df(self.cy, 0.0, 1.0),
            rx=df(self.rx, 0.0, 0.5),
            ry=df(self.ry, 0.0, 0.5),
            theta=df(self.theta, 0.0, math.pi),
            r=di(self.r, 0, 255),
            g=di(self.g, 0, 255),
            b=di(self.b, 0, 255),
            a=df(self.a, 0.0, 1.0),
            active=self.active,
        )
        # Flip active con probabilidad 5%
        if random.random() < 0.05:
            result.active = not result.active
        return result

    def blend(self, other: "EllipseGene", alpha: float) -> "EllipseGene":
        """Interpola atributos entre este gen y otro: alpha * self + (1-alpha) * other."""
        def lerp_float(a: float, b: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, alpha * a + (1 - alpha) * b))

        def lerp_int(a: int, b: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, round(alpha * a + (1 - alpha) * b)))

        return EllipseGene(
            cx=lerp_float(self.cx, other.cx, 0.0, 1.0),
            cy=lerp_float(self.cy, other.cy, 0.0, 1.0),
            rx=lerp_float(self.rx, other.rx, 0.0, 0.5),
            ry=lerp_float(self.ry, other.ry, 0.0, 0.5),
            theta=lerp_float(self.theta, other.theta, 0.0, math.pi),
            r=lerp_int(self.r, other.r, 0, 255),
            g=lerp_int(self.g, other.g, 0, 255),
            b=lerp_int(self.b, other.b, 0, 255),
            a=lerp_float(self.a, other.a, 0.0, 1.0),
            active=self.active,
        )

    def mutate_gaussian(self, sigma: float) -> "EllipseGene":
        """Perturba cada atributo sumando delta ~ N(0, sigma), clampeando al rango valido."""
        def gauss_float(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val + random.gauss(0, sigma)))

        def gauss_int(val: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, round(val + random.gauss(0, sigma * 255))))

        return EllipseGene(
            cx=gauss_float(self.cx, 0.0, 1.0),
            cy=gauss_float(self.cy, 0.0, 1.0),
            rx=gauss_float(self.rx, 0.0, 0.5),
            ry=gauss_float(self.ry, 0.0, 0.5),
            theta=gauss_float(self.theta, 0.0, math.pi),
            r=gauss_int(self.r, 0, 255),
            g=gauss_int(self.g, 0, 255),
            b=gauss_int(self.b, 0, 255),
            a=gauss_float(self.a, 0.0, 1.0),
            active=self.active,
        )

    def to_dict(self) -> dict:
        """Serializa el gen a un diccionario."""
        return {
            "type": "ellipse",
            "cx": self.cx, "cy": self.cy,
            "rx": self.rx, "ry": self.ry,
            "theta": self.theta,
            "r": self.r, "g": self.g, "b": self.b,
            "a": self.a, "active": self.active,
        }

    def to_row(self) -> np.ndarray:
        """Convierte a fila numpy (11,) con padding en indice 5."""
        return np.array([
            self.cx, self.cy, self.rx, self.ry, self.theta, 0.0,
            float(self.r), float(self.g), float(self.b), self.a,
            1.0 if self.active else 0.0,
        ], dtype=np.float64)

    @classmethod
    def from_row(cls, row: np.ndarray) -> "EllipseGene":
        """Crea un EllipseGene desde una fila numpy (11,)."""
        return cls(
            cx=float(row[0]), cy=float(row[1]),
            rx=float(row[2]), ry=float(row[3]),
            theta=float(row[4]),
            r=int(row[6]), g=int(row[7]), b=int(row[8]),
            a=float(row[9]),
            active=bool(row[10] > 0.5),
        )
