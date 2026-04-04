import random
from dataclasses import dataclass
from genes.gene import Gene


@dataclass
class TriangleGene(Gene):
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
        )

    def copy(self) -> "TriangleGene":
        """Retorna una copia independiente. Como todos los atributos son primitivos, basta con crear una nueva instancia."""
        return TriangleGene(
            x1=self.x1, y1=self.y1,
            x2=self.x2, y2=self.y2,
            x3=self.x3, y3=self.y3,
            r=self.r, g=self.g, b=self.b, a=self.a,
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

    def to_dict(self) -> dict:
        """Serializa el gen a un diccionario."""
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "x3": self.x3, "y3": self.y3,
            "r": self.r, "g": self.g, "b": self.b,
            "a": self.a,
        }
