#!/bin/bash
# =============================================================
# Experimento Formas: Triángulo vs Elipse
# Corre el experimento, genera plots, y renombra para las slides
# =============================================================
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  PASO 1/3: Corriendo experimento formas"
echo "  2 configs × 4 imgs × 3 seeds = 24 runs"
echo "============================================"
time python3 experiments/exp_formas.py

echo ""
echo "============================================"
echo "  PASO 2/3: Generando plots por imagen"
echo "============================================"

RESULTS="experiments/results/formas"
PLOTS="experiments/plots/formas"

# plot_results.py con --name custom para que el prefijo coincida con las slides
for img in 1_Ucrania 2_Grecia 3_Apple 4_Cubista; do
    echo "--- Plots para $img ---"
    python3 experiments/plot_results.py \
        --input "$RESULTS/$img" \
        --output "$PLOTS/$img" \
        --name "formas_$img"
done

echo ""
echo "============================================"
echo "  PASO 3/3: Generando plots cruzados"
echo "============================================"
python3 experiments/plot_cross_image.py \
    --input "$RESULTS" \
    --output "$PLOTS" \
    --name formas

echo ""
echo "============================================"
echo "  LISTO! Verificando archivos generados:"
echo "============================================"

# Verificar que existen los archivos que esperan las slides
EXPECTED=(
    "$PLOTS/1_Ucrania/formas_1_Ucrania_fitness_best.png"
    "$PLOTS/1_Ucrania/formas_1_Ucrania_image_grid.png"
    "$PLOTS/2_Grecia/formas_2_Grecia_fitness_best.png"
    "$PLOTS/2_Grecia/formas_2_Grecia_image_grid.png"
    "$PLOTS/3_Apple/formas_3_Apple_fitness_best.png"
    "$PLOTS/3_Apple/formas_3_Apple_image_grid.png"
    "$PLOTS/4_Cubista/formas_4_Cubista_fitness_best.png"
    "$PLOTS/4_Cubista/formas_4_Cubista_image_grid.png"
    "$PLOTS/formas_heatmap_fitness.png"
    "$PLOTS/formas_barras_comparacion.png"
)

ok=0
fail=0
for f in "${EXPECTED[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
        ((ok++))
    else
        echo "  ✗ FALTA: $f"
        ((fail++))
    fi
done

echo ""
echo "Resultado: $ok OK, $fail faltantes de ${#EXPECTED[@]} esperados"
