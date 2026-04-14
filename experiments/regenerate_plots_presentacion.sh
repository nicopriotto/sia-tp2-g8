#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_TRI_CONFIGS="10_triangulos,50_triangulos,200_triangulos"
SOURCE_IMAGES="1_Ucrania,3_Apple,4_Cubista"
IMAGE_RENAMES="1_Ucrania:1_Ucrania,3_Apple:2_Apple,4_Cubista:3_Cubista"
COMPLEJIDAD_INPUT_IMAGES="1_Ucrania,3_Apple,4_Cubista"

# Mapa source -> alias para salida de plots por imagen.
declare -A IMAGE_ALIAS=(
  ["1_Ucrania"]="1_Ucrania"
  ["3_Apple"]="2_Apple"
  ["4_Cubista"]="3_Cubista"
)

echo "=== Regenerando plots para presentacion (Ucrania, Apple, Cubista) ==="

echo "Limpiando plots viejos de experimentos disponibles..."
while IFS= read -r exp; do
  rm -rf "experiments/plots/$exp"
done < <(find experiments/results -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

echo ""
echo "Ploteando complejidad (subset 1_Ucrania, 2_Apple, 3_Cubista)..."
if [ -d "experiments/results/complejidad" ]; then
  .venv/bin/python experiments/plot_results.py \
    --input experiments/results/complejidad \
    --output experiments/plots/complejidad \
    --name "complejidad - 1_Ucrania_2_Apple_3_Cubista" \
    --avg_mode error_log \
    --include_configs "$COMPLEJIDAD_INPUT_IMAGES"
fi

echo ""
echo "Ploteando experimentos por imagen (reindexados a 1/2/3)..."
while IFS= read -r exp; do
  [ "$exp" = "complejidad" ] && continue

  for src_img in 1_Ucrania 3_Apple 4_Cubista; do
    input_dir="experiments/results/$exp/$src_img"
    [ -d "$input_dir" ] || continue

    alias_img="${IMAGE_ALIAS[$src_img]}"
    output_dir="experiments/plots/$exp/$alias_img"
    plot_name="$exp - $alias_img"

    extra_args=(--avg_mode error_log)

    if [ "$exp" = "num_triangulos" ]; then
      extra_args+=(--include_configs "$NUM_TRI_CONFIGS")
    fi

    # Mantener visuales solicitadas previamente.
    if [ "$exp" = "seleccion" ]; then
      extra_args+=(--avg_small_multiples --avg_gap_to_best)
    fi

    echo "  $exp/$src_img -> $alias_img"
    .venv/bin/python experiments/plot_results.py \
      --input "$input_dir" \
      --output "$output_dir" \
      --name "$plot_name" \
      "${extra_args[@]}"
  done
done < <(find experiments/results -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

echo ""
echo "Regenerando analisis cruzado (con reindexado de imagenes)..."
.venv/bin/python experiments/plot_cross_image.py \
  --all \
  --input experiments/results \
  --output experiments/plots \
  --include_images "$SOURCE_IMAGES" \
  --image_renames "$IMAGE_RENAMES"

# Re-generar num_triangulos filtrado para evitar que entren configs viejas.
if [ -d "experiments/results/num_triangulos" ]; then
  .venv/bin/python experiments/plot_cross_image.py \
    --input experiments/results/num_triangulos \
    --output experiments/plots/num_triangulos \
    --name num_triangulos \
    --include_configs "$NUM_TRI_CONFIGS" \
    --include_images "$SOURCE_IMAGES" \
    --image_renames "$IMAGE_RENAMES"
fi

echo ""
echo "Listo. Plots regenerados en experiments/plots"
