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

### Campos soportados por el JSON de entrada (completo)

#### 1) Campos obligatorios (parser)

| Campo | Tipo | Descripcion |
|---|---|---|
| `triangle_count` | int | Cantidad de genes (figuras) por individuo. |
| `population_size` | int | Tamaño de poblacion. |
| `max_generations` | int | Tope de generaciones. |
| `fitness_threshold` | float | Umbral de fitness para corte por objetivo. |
| `crossover_probability` | float | Probabilidad de aplicar crossover. |
| `mutation_rate` | float | Probabilidad de mutar. |
| `survival_strategy` | str | Estrategia de supervivencia (`Aditiva`, `Exclusiva`). |
| `fitness_function` | str | Funcion de fitness. |
| `k_offspring` | int | Cantidad de hijos por generacion. |
| `save_every` | int | Frecuencia de snapshots `gen_XXXX.png`. |

#### 2) Campos de operadores y criterios generales

| Campo | Tipo | Default | Descripcion |
|---|---|---|---|
| `selection_method` | str o list | `Elite` | Metodo de seleccion (formato simple o lista legacy). |
| `selection_methods` | list | `[]` | Lista de metodos de seleccion (tiene prioridad sobre `selection_method`). |
| `crossover_methods` | list | `["OnePoint"]` | Lista de operadores de crossover. |
| `mutation_methods` | list | `["Gen"]` | Lista de operadores de mutacion. |
| `boltzmann_t0` | float | `100.0` | Temperatura inicial para Boltzmann. |
| `boltzmann_tc` | float | `1.0` | Temperatura minima para Boltzmann. |
| `boltzmann_k` | float | `0.01` | Decaimiento de temperatura Boltzmann. |
| `tournament_m` | int | `5` | Tamaño de torneo deterministico. |
| `tournament_threshold` | float | `0.75` | Umbral del torneo probabilistico. |
| `non_uniform_b` | float | `1.0` | Parametro de decaimiento para mutacion no uniforme. |
| `generational_gap` | float | `1.0` | Fraccion renovada por generacion. Debe estar en `[0,1]`. |
| `max_seconds` | float | `0.0` | Corte por tiempo (0 desactiva). |
| `content_threshold` | float | `0.0` | Umbral para criterio de contenido/estancamiento. |
| `content_generations` | int | `0` | Ventana de generaciones para criterio de contenido. |
| `structure_threshold` | float | `0.0` | Umbral para criterio de estructura/convergencia. |
| `structure_generations` | int | `0` | Ventana para criterio de estructura. |
| `min_error` | float | `0.0` | Corte por error minimo (0 desactiva). |

#### 3) Campos de render y forma de gen

| Campo | Tipo | Default | Descripcion |
|---|---|---|---|
| `use_gpu` | bool | `false` | Activa renderer GPU si hay soporte. |
| `gpu_device` | str | `auto` | `auto`, `dedicated`, `integrated`. |
| `gene_type` | str | `triangle` | Tipo de gen (`triangle` o `ellipse`). |

Notas de GPU:

- `gpu_device` acepta `auto`, `dedicated` o `integrated`.
- Si se configura `gpu_device: "dedicated"` y no hay GPU dedicada disponible, la ejecucion falla con error (no hace fallback silencioso a CPU).

#### 4) Campos de operadores avanzados

| Campo | Tipo | Default | Descripcion |
|---|---|---|---|
| `arithmetic_alpha` | float | `0.5` | Alpha del crossover aritmetico. |
| `gaussian_sigma` | float | `0.1` | Sigma geometrica para mutacion gaussiana. |
| `gaussian_sigma_color` | float | `0.1` | Sigma de color/alpha (si falta, hereda `gaussian_sigma`). |
| `gaussian_decay_b` | float | `0.0` | Decay de sigma por progreso (`0` sin decay). |
| `gaussian_swap_rate` | float | `0.0` | Probabilidad de swap de capas (z-index). |
| `smart_init` | bool | `false` | Inicializacion inteligente segun imagen objetivo. |
| `elite_count` | int | `1` | Cantidad de elites usadas por operadores que lo requieran. |
| `adaptive_operator_weights` | bool | `false` | Activa ajuste adaptativo de pesos en seleccion/mutacion. |
| `adaptive_operator_delta` | float | `0.05` | Delta de premio/castigo de pesos adaptativos (> 0). |

#### 5) Campos de anti-estancamiento

| Campo | Tipo | Default | Descripcion |
|---|---|---|---|
| `stagnation_check_interval` | int | `0` | Cada cuantas generaciones evaluar estancamiento (`0` desactiva). |
| `stagnation_threshold` | float | `1e-5` | Mejora minima para no considerar estancamiento. |
| `stagnation_mutation_boost` | float | `3.0` | Multiplicador de mutacion al detectar estancamiento. |
| `stagnation_replace_pct` | float | `0.2` | Fraccion de poblacion a reemplazar al estancarse. |

#### 6) Campos de Island Model

| Campo | Tipo | Default | Descripcion |
|---|---|---|---|
| `island_enabled` | bool | `false` | Activa modo multi-isla. |
| `island_count` | int | `5` | Cantidad de islas (si `island_enabled=true`, debe ser `>= 2`). |
| `island_migration_interval` | int | `50` | Intervalo de migracion (`>= 1`). |
| `island_migration_count` | int | `2` | Migrantes por evento (`>=1` y `< population_size`). |
| `island_topology` | str | `ring` | Topologia (`ring`, `fully_connected`). |
| `island_configs` | list[dict] | `[]` | Overrides por isla; si se usa, su largo debe coincidir con `island_count`. |

Cada elemento de `island_configs` puede incluir cualquier campo de `Config` excepto los globales del island model (`island_enabled`, `island_count`, `island_migration_interval`, `island_migration_count`, `island_topology`, `island_configs`, `max_generations`, `fitness_threshold`). Tambien acepta `name` solo para logging.

#### 7) Validos por categoria

- Seleccion: `Elite`, `Ruleta`, `Universal`, `Ranking`, `Boltzmann`, `TorneosDeterministicos`, `TorneosProbabilisticos`
- Crossover: `OnePoint`, `TwoPoint`, `Uniform`, `Annular`, `Aritmetico`
- Mutacion: `Gen`, `MultiGen`, `Uniforme`, `Completa`, `NoUniforme`, `Gaussiana`
- Supervivencia: `Aditiva`, `Exclusiva`
- Fitness: `MSE`, `MAE`, `GMSD`, `Oklab`, `MSSSIM`, `FSIM`, `SSIM`, `LinearMSE`

#### 8) Formato ponderado para listas de operadores

Los campos `selection_methods`, `crossover_methods` y `mutation_methods` aceptan:

- strings (peso uniforme)
- objetos `{ "method": "...", "weight": ... }` con peso positivo

Ejemplo:

```json
{
  "selection_methods": [
    {"method": "Boltzmann", "weight": 2.0},
    {"method": "Ranking", "weight": 1.0}
  ],
  "crossover_methods": [
    {"method": "Uniform", "weight": 3.0},
    {"method": "TwoPoint", "weight": 1.0}
  ],
  "mutation_methods": [
    {"method": "Gaussiana", "weight": 4.0},
    {"method": "MultiGen", "weight": 1.0}
  ]
}
```

Nota: `selection_weights`, `crossover_weights` y `mutation_weights` son campos internos del objeto `Config`; se calculan automaticamente y no hace falta declararlos en el JSON.

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
