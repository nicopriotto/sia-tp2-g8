import random
from dataclasses import dataclass
from genes.polygon_gene import PolygonGene


@dataclass
class TriangleGene(PolygonGene):
    x1: float  # [0, 1]
    y1: float  # [0, 1]
    x2: float  # [0, 1]
    y2: float  # [0, 1]
    x3: float  # [0, 1]
    y3: float  # [0, 1]
    r: int     # [0, 255]
    g: int     # [0, 255]
    b: int     # [0, 255]
    a: float   # [0, 1]
    active: bool = True  # gen activo o inactivo (triangulo se dibuja o no)

    @classmethod
    def random(cls) -> "TriangleGene":
        """Crea un TriangleGene con valores aleatorios dentro de los rangos validos."""
        return cls(
            x1=random.random(), y1=random.random(),
            x2=random.random(), y2=random.random(),
            x3=random.random(), y3=random.random(),
            r=random.randint(0, 255),
            g=random.randint(0, 255),
            b=random.randint(0, 255),
            a=random.random(),
            active=True,
        )

    def copy(self) -> "TriangleGene":
        """Retorna una copia independiente. Como todos los atributos son primitivos, basta con crear una nueva instancia."""
        return TriangleGene(
            x1=self.x1, y1=self.y1,
            x2=self.x2, y2=self.y2,
            x3=self.x3, y3=self.y3,
            r=self.r, g=self.g, b=self.b, a=self.a,
            active=self.active,
        )

    def mutate_replace(self) -> "TriangleGene":
        """Genera un gen completamente nuevo con valores aleatorios."""
        return TriangleGene.random()

    def mutate_delta(self, strength: float) -> "TriangleGene":
        """
        Perturba cada atributo sumando un delta aleatorio proporcional a strength.
        - Coords: delta en [-strength, +strength], clamp a [0, 1]
        - RGB: delta en [-strength*255, +strength*255], clamp a [0, 255], redondear a int
        - Alpha: delta en [-strength, +strength], clamp a [0, 1]

        Si el resultado es degenerado, reintenta con strength*2 hasta 10 veces.
        Si todos los intentos fallan, retorna una copia del gen original.
        """
        current_strength = strength
        for _ in range(10):
            candidate = self._apply_delta(current_strength)
            if not candidate._is_degenerate():
                # Con probabilidad 5%, flipear active
                if random.random() < 0.05:
                    candidate.active = not candidate.active
                return candidate
            current_strength *= 2
        return self.copy()

    def _apply_delta(self, strength: float) -> "TriangleGene":
        """Aplica perturbacion delta a todos los atributos."""
        def delta_float(value: float, low: float, high: float) -> float:
            d = random.uniform(-strength, strength)
            return max(low, min(high, value + d))

        def delta_int(value: int, low: int, high: int) -> int:
            d = random.uniform(-strength * 255, strength * 255)
            return max(low, min(high, round(value + d)))

        return TriangleGene(
            x1=delta_float(self.x1, 0.0, 1.0),
            y1=delta_float(self.y1, 0.0, 1.0),
            x2=delta_float(self.x2, 0.0, 1.0),
            y2=delta_float(self.y2, 0.0, 1.0),
            x3=delta_float(self.x3, 0.0, 1.0),
            y3=delta_float(self.y3, 0.0, 1.0),
            r=delta_int(self.r, 0, 255),
            g=delta_int(self.g, 0, 255),
            b=delta_int(self.b, 0, 255),
            a=delta_float(self.a, 0.0, 1.0),
            active=self.active,
        )

    def _is_degenerate(self, epsilon: float = 1e-10) -> bool:
        """
        Retorna True si el triangulo es degenerado (area ~= 0).
        Formula del area: |x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)| / 2
        """
        area = abs(
            self.x1 * (self.y2 - self.y3)
            + self.x2 * (self.y3 - self.y1)
            + self.x3 * (self.y1 - self.y2)
        ) / 2.0
        return area < epsilon

    def blend(self, other: "TriangleGene", alpha: float) -> "TriangleGene":
        """Interpola atributos entre este gen y otro: alpha * self + (1-alpha) * other."""
        def lerp_float(a: float, b: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, alpha * a + (1 - alpha) * b))

        def lerp_int(a: int, b: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, round(alpha * a + (1 - alpha) * b)))

        return TriangleGene(
            x1=lerp_float(self.x1, other.x1, 0.0, 1.0),
            y1=lerp_float(self.y1, other.y1, 0.0, 1.0),
            x2=lerp_float(self.x2, other.x2, 0.0, 1.0),
            y2=lerp_float(self.y2, other.y2, 0.0, 1.0),
            x3=lerp_float(self.x3, other.x3, 0.0, 1.0),
            y3=lerp_float(self.y3, other.y3, 0.0, 1.0),
            r=lerp_int(self.r, other.r, 0, 255),
            g=lerp_int(self.g, other.g, 0, 255),
            b=lerp_int(self.b, other.b, 0, 255),
            a=lerp_float(self.a, other.a, 0.0, 1.0),
            active=self.active,
        )

    def mutate_gaussian(self, sigma: float) -> "TriangleGene":
        """Perturba cada atributo sumando delta ~ N(0, sigma), clampeando al rango valido."""
        def gauss_float(val: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, val + random.gauss(0, sigma)))

        def gauss_int(val: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, round(val + random.gauss(0, sigma * 255))))

        return TriangleGene(
            x1=gauss_float(self.x1, 0.0, 1.0),
            y1=gauss_float(self.y1, 0.0, 1.0),
            x2=gauss_float(self.x2, 0.0, 1.0),
            y2=gauss_float(self.y2, 0.0, 1.0),
            x3=gauss_float(self.x3, 0.0, 1.0),
            y3=gauss_float(self.y3, 0.0, 1.0),
            r=gauss_int(self.r, 0, 255),
            g=gauss_int(self.g, 0, 255),
            b=gauss_int(self.b, 0, 255),
            a=gauss_float(self.a, 0.0, 1.0),
            active=self.active,
        )

    def to_dict(self) -> dict:
        """Serializa el gen a un diccionario."""
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "x3": self.x3, "y3": self.y3,
            "r": self.r, "g": self.g, "b": self.b,
            "a": self.a,
            "active": self.active,
        }
