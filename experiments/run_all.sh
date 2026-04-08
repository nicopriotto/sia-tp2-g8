#!/bin/bash
set -e
cd "$(dirname "$0")/.."

export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia

echo "=== Experimentos TP2 - $(date) ==="
echo ""

echo "[1/12] Complejidad de imagenes..."
.venv/bin/python experiments/exp_complejidad.py

echo "[2/12] Cantidad de triangulos..."
.venv/bin/python experiments/exp_num_triangulos.py

echo "[3/12] Funciones de fitness..."
.venv/bin/python experiments/exp_fitness.py

echo "[4/12] Inicializacion de poblacion..."
.venv/bin/python experiments/exp_inicializacion.py

echo "[5/12] Metodos de seleccion..."
.venv/bin/python experiments/exp_seleccion.py

echo "[6/12] Operadores de crossover..."
.venv/bin/python experiments/exp_crossover.py

echo "[7/12] Operadores de mutacion..."
.venv/bin/python experiments/exp_mutacion.py

echo "[8/12] Estrategias de supervivencia..."
.venv/bin/python experiments/exp_supervivencia.py

echo "[9/12] Condiciones de corte..."
.venv/bin/python experiments/exp_corte.py

echo "[10/12] Anti-estancamiento..."
.venv/bin/python experiments/exp_anti_estancamiento.py

echo "[11/12] CPU vs GPU..."
.venv/bin/python experiments/exp_gpu.py

echo "[12/12] Triangulos vs elipses..."
.venv/bin/python experiments/exp_formas.py

echo ""
echo "=== Generando plots por imagen ==="

# Complejidad: estructura plana (compara imagenes entre si)
echo "  Ploteando complejidad..."
.venv/bin/python experiments/plot_results.py --input experiments/results/complejidad --output experiments/plots/complejidad

# Resto: estructura por imagen, plotear cada sub-carpeta
EXPERIMENTS="num_triangulos fitness inicializacion seleccion crossover mutacion supervivencia corte anti_estancamiento formas"

for exp in $EXPERIMENTS; do
    for img_dir in experiments/results/$exp/*/; do
        if [ -d "$img_dir" ]; then
            img_name=$(basename "$img_dir")
            echo "  Ploteando $exp/$img_name..."
            .venv/bin/python experiments/plot_results.py \
                --input "$img_dir" \
                --output "experiments/plots/$exp/$img_name" \
                --name "$exp - $img_name"
        fi
    done
done

echo ""
echo "=== Generando analisis cruzado entre imagenes ==="
.venv/bin/python experiments/plot_cross_image.py \
    --all \
    --input experiments/results \
    --output experiments/plots

echo ""
echo "=== Ploteando islas noche estrellada ==="
.venv/bin/python experiments/plot_islas_noche.py

echo ""
echo "=== Listo! $(date) ==="
echo ""
echo "Estructura de plots:"
echo "  experiments/plots/complejidad/             - 4 imagenes comparadas"
echo "  experiments/plots/<exp>/<imagen>/           - graficos por imagen"
echo "  experiments/plots/<exp>/heatmap_fitness.png - analisis cruzado"
echo "  experiments/plots/<exp>/barras_comparacion.png"
echo "  experiments/plots/<exp>/heatmap_convergencia.png"
echo "  experiments/plots/gpu/<imagen>/             - CPU vs GPU"
echo "  experiments/plots/islas_noche/              - overnight run"
