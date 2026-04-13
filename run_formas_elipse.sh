#!/bin/bash
# Re-corre solo las elipses (ahora con CPU renderer) y regenera plots
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  PASO 1/3: Corriendo solo Elipse (CPU)"
echo "  4 imgs × 3 seeds = 12 runs"
echo "============================================"
time python3 -c "
import sys, os
sys.path.insert(0, '.')
from experiments.run_experiment import run_experiment_all_images

run_experiment_all_images(
    name='formas_elipse',
    configs=[('Elipse', {'gene_type': 'ellipse'})],
    output_base='experiments/results/formas',
    max_generations=1000,
)
"

echo ""
echo "============================================"
echo "  PASO 2/3: Regenerando plots por imagen"
echo "============================================"
RESULTS="experiments/results/formas"
PLOTS="experiments/plots/formas"

for img in 1_Ucrania 2_Grecia 3_Apple 4_Cubista; do
    echo "--- Plots para $img ---"
    python3 experiments/plot_results.py \
        --input "$RESULTS/$img" \
        --output "$PLOTS/$img" \
        --name "formas_$img"
done

echo ""
echo "============================================"
echo "  PASO 3/3: Regenerando plots cruzados"
echo "============================================"
python3 experiments/plot_cross_image.py \
    --input "$RESULTS" \
    --output "$PLOTS" \
    --name formas

echo ""
echo "LISTO! Revisa experiments/plots/formas/"
