"""
auto_config.py — Analiza la complejidad de una imagen y lanza el GA
con la configuración adecuada (bajo / medio / alto).

Uso:
    python auto_config.py <ruta_imagen>
"""

import sys
import os
import logging

from analysis.image_complexity import image_complexity
from main import run_from_paths

# Umbrales según la fórmula: C = α·C_color + (1-α)·C_forma, α=0.5
_UMBRAL_BAJO  = 0.20   # C < 0.20  → bajo
_UMBRAL_MEDIO = 0.45   # C < 0.45  → medio ; C ≥ 0.45 → alto

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "run_configs")

_CONFIGS = {
    "bajo":  os.path.join(_CONFIG_DIR, "bajo.json"),
    "medio": os.path.join(_CONFIG_DIR, "medio.json"),
    "alto":  os.path.join(_CONFIG_DIR, "alto.json"),
}


def clasificar(C: float) -> str:
    if C < _UMBRAL_BAJO:
        return "bajo"
    elif C < _UMBRAL_MEDIO:
        return "medio"
    else:
        return "alto"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if not args:
        print("Uso: python auto_config.py <ruta_imagen>")
        return 1

    image_path = args[0]

    if not os.path.isfile(image_path):
        print(f"Error: no se encontró el archivo '{image_path}'")
        return 1

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    # --- Análisis de complejidad ---
    print("\nAnalizando complejidad de la imagen...")
    resultado = image_complexity(image_path)

    C       = resultado["C"]
    C_color = resultado["C_color"]
    C_forma = resultado["C_forma"]

    nivel = clasificar(C)

    print(f"\n=== Resultado del análisis ===")
    print(f"  Complejidad total  C       = {C:.4f}")
    print(f"  Componente color   C_color = {C_color:.4f}")
    print(f"  Componente forma   C_forma = {C_forma:.4f}")
    print(f"\n  Clasificación: {nivel.upper()}")

    if nivel == "bajo":
        print(f"  (C = {C:.4f} < {_UMBRAL_BAJO}  → configuración BAJA)")
    elif nivel == "medio":
        print(f"  (C = {C:.4f} en [{_UMBRAL_BAJO}, {_UMBRAL_MEDIO}) → configuración MEDIA)")
    else:
        print(f"  (C = {C:.4f} ≥ {_UMBRAL_MEDIO} → configuración ALTA)")

    config_path = _CONFIGS[nivel]
    print(f"\nUsando configuración: {config_path}")
    print("=" * 40 + "\n")

    # --- Ejecución del GA ---
    run_from_paths(image_path, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
