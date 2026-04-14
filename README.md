# Algoritmo Genetico para Aproximacion de Imagenes

## Descripcion

Implementacion de un algoritmo genetico para aproximar una imagen objetivo usando figuras RGBA semitransparentes (triangulos o elipses) sobre un canvas blanco.

El pipeline soporta:
- multiples metodos de seleccion,
- multiples operadores de crossover y mutacion,
- estrategias de supervivencia,
- funciones de fitness intercambiables,
- renderer CPU o GPU.

## Requisitos

- Python 3.10+
- Dependencias de Python en `requirements.txt`

## Instalacion

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para graficos de experimentos:

```bash
pip install matplotlib pandas seaborn Pillow
```

## Ejecucion basica

```bash
python3 main.py <ruta_imagen> <ruta_config_json>
```

Parametros de entrada:

- `<ruta_imagen>`: path a la imagen objetivo (por ejemplo `images/1.jpg`).
- `<ruta_config_json>`: path al archivo JSON de configuracion (por ejemplo `run_configs/config.json`).

Ejemplo real del repo:

```bash
python3 main.py images/1.jpg run_configs/config.json
```

## Auto-configuracion segun complejidad

`auto_config.py` analiza automaticamente la complejidad de la imagen y selecciona la configuracion optima antes de lanzar el GA.

```bash
python3 auto_config.py <ruta_imagen>
```

El script calcula la complejidad de la imagen usando la formula:

```
C = alpha * C_color + (1 - alpha) * C_forma    (alpha = 0.5)
```

Donde:
- `C_color`: entropia del histograma de color RGB (diversidad cromatica).
- `C_forma`: promedio ponderado por area de `(1 - solidez)` sobre regiones conectadas (irregularidad de formas).

Segun el valor de `C` obtenido, se selecciona automaticamente una configuracion:

| Clasificacion | Umbral       | Config usada            | Triangulos | Generaciones |
|---------------|--------------|-------------------------|------------|--------------|
| Baja          | C < 0.20     | `run_configs/bajo.json` | 50         | 500          |
| Media         | 0.20 <= C < 0.45 | `run_configs/medio.json` | 100    | 1000         |
| Alta          | C >= 0.45    | `run_configs/alto.json` | 200        | 3000         |

El script imprime el desglose completo antes de iniciar la ejecucion:

```
Analizando complejidad de la imagen...

=== Resultado del analisis ===
  Complejidad total  C       = 0.3412
  Componente color   C_color = 0.4100
  Componente forma   C_forma = 0.2724

  Clasificacion: MEDIO
  (C = 0.3412 en [0.20, 0.45) -> configuracion MEDIA)

Usando configuracion: run_configs/medio.json
```

## Configuracion

La configuracion se carga desde JSON. Como punto de partida, usar:

- `run_configs/base-config.json`

Campos principales:

- `triangle_count`, `population_size`, `max_generations`, `fitness_threshold`
- `selection_method` o `selection_methods`
- `crossover_methods`, `crossover_probability`
- `mutation_methods`, `mutation_rate`
- `survival_strategy`, `fitness_function`
- `k_offspring`, `save_every`
- `use_gpu`, `gpu_device`

Notas de GPU:

- `gpu_device` acepta `auto`, `dedicated` o `integrated`.
- Si se configura `gpu_device: "dedicated"` y no hay GPU dedicada disponible, la ejecucion falla con error (no hace fallback silencioso a CPU).

Metodos soportados:

- Seleccion: `Elite`, `Ruleta`, `Universal`, `Ranking`, `Boltzmann`, `TorneosDeterministicos`, `TorneosProbabilisticos`
- Crossover: `OnePoint`, `TwoPoint`, `Uniform`, `Annular`, `Aritmetico`
- Mutacion: `Gen`, `MultiGen`, `Uniforme`, `Completa`, `NoUniforme`, `Gaussiana`
- Supervivencia: `Aditiva`, `Exclusiva`
- Fitness: `MSE`, `MAE`, `GMSD`, `Oklab`, `MSSSIM`, `FSIM`, `SSIM`, `LinearMSE`

Formato ponderado (opcional) para operadores:

```json
{
  "crossover_methods": [
    {"method": "OnePoint", "weight": 3.0},
    {"method": "Uniform", "weight": 1.0}
  ]
}
```

## Archivos de salida

En una corrida normal se generan en `output/`:

- `output/final.png`
- `output/triangles.json`
- `output/metrics.csv`
- `output/gen_XXXX.png` (si `save_every > 0`)

## Experimentos

Scripts disponibles en `experiments/`:

- `exp_complejidad.py`
- `exp_fitness.py`
- `exp_inicializacion.py`
- `exp_seleccion.py`
- `exp_crossover.py`
- `exp_mutacion.py`
- `exp_supervivencia.py`
- `exp_corte.py`
- `exp_anti_estancamiento.py`
- `exp_formas.py`
- `exp_gpu.py`
- `exp_num_triangulos.py` (multi-imagen)
- `exp_triangulos.py` (imagen individual)
- `exp_pc.py`
- `exp_ponderado.py`

Ejemplo:

```bash
.venv/bin/python experiments/exp_seleccion.py
```

Opciones comunes que aceptan estos scripts:

- `--max_generations`
- `--seeds`
- `--quick`
- `--output`

## Generacion de graficos

Graficos por imagen/config:

```bash
.venv/bin/python experiments/plot_results.py \
  --input experiments/results/seleccion/1_Ucrania \
  --output experiments/plots/seleccion/1_Ucrania \
  --name "seleccion - 1_Ucrania" \
  --avg_mode error_log
```

Resumen cruzado entre imagenes:

```bash
.venv/bin/python experiments/plot_cross_image.py \
  --input experiments/results/seleccion \
  --output experiments/plots/seleccion \
  --name seleccion
```

## Batch runners

Runner general por campanias:

```bash
.venv/bin/python experiments/run_batch_tp.py \
  --images images/1.jpg images/3.jpg \
  --base-config run_configs/config.json \
  --seed 42
```

Runner chico (solo seleccion):

```bash
.venv/bin/python experiments/run_batch_selection_only.py \
  --images images/1.jpg \
  --base-config run_configs/config.json \
  --seed 42
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Estructura del proyecto

```text
config/          loader y validacion de configuracion
core/            algoritmo genetico, poblacion, metricas, island model
genes/           representacion de genes (triangle/ellipse)
selection/       operadores de seleccion
crossover/       operadores de cruza
mutation/        operadores de mutacion
survival/        estrategias de supervivencia
fitness/         funciones de fitness
render/          render CPU/GPU y shaders
experiments/     ejecucion de experimentos y generacion de plots
run_configs/     configuraciones JSON de ejemplo
output/          salida de corridas
tests/           tests unitarios e integracion
```

## Autores

Grupo 8 - Sistemas de Inteligencia Artificial - ITBA
