# Algoritmo Genetico para Aproximacion de Imagenes con Figuras Geometricas

## Descripcion

Proyecto de la materia Sistemas de Inteligencia Artificial (SIA) que implementa un algoritmo genetico para aproximar imagenes usando triangulos y elipses RGBA semi-transparentes sobre un canvas blanco. El sistema soporta multiples operadores de seleccion, crossover, mutacion y supervivencia, todos configurables via un archivo JSON.

## Requisitos

- Python 3.10 o superior
- Dependencias: ver `requirements.txt`

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd sia-tp2-g8

# Instalar dependencias
pip install -r requirements.txt
```

Para experimentacion con graficos (opcional):

```bash
pip install matplotlib pandas
```

## Ejecucion

```bash
python3 main.py <ruta_imagen> <ruta_config>
```

Ejemplo:

```bash
python3 main.py images/mona_lisa.png config.json
```

El programa carga la imagen objetivo, ejecuta el algoritmo genetico con la configuracion dada, y guarda los resultados en la carpeta `output/`.

## Configuracion

El archivo de configuracion es un JSON con los siguientes campos:

### Campos obligatorios

| Campo | Tipo | Rango/Valores | Descripcion |
|---|---|---|---|
| `triangle_count` | int | >= 1 | Cantidad maxima de figuras geometricas (triangulos o elipses) |
| `population_size` | int | >= 2 | Tamano de la poblacion |
| `max_generations` | int | >= 1 | Maximo de generaciones |
| `fitness_threshold` | float | (0, 1] | Fitness objetivo para criterio de corte |
| `selection_method` | str | ver metodos abajo | Metodo de seleccion principal (formato simple) |
| `crossover_methods` | list | ver metodos abajo | Metodos de crossover (soporta pesos) |
| `crossover_probability` | float | [0, 1] | Probabilidad de aplicar crossover a cada par |
| `mutation_methods` | list | ver metodos abajo | Metodos de mutacion (soporta pesos) |
| `mutation_rate` | float | [0, 1] | Probabilidad de mutar cada gen |
| `survival_strategy` | str | "Aditiva", "Exclusiva" | Estrategia de supervivencia |
| `fitness_function` | str | "MSE", "MAE" | Funcion de fitness |
| `k_offspring` | int | >= 2 | Cantidad de hijos a generar por generacion |
| `save_every` | int | >= 0 | Frecuencia de snapshots intermedios (0 = desactivado) |

### Campos opcionales con defaults

| Campo | Tipo | Default | Rango/Valores | Descripcion |
|---|---|---|---|---|
| `boltzmann_t0` | float | 100.0 | > 0 | Temperatura inicial para seleccion Boltzmann |
| `boltzmann_tc` | float | 1.0 | > 0 | Temperatura minima para seleccion Boltzmann |
| `boltzmann_k` | float | 0.01 | > 0 | Constante de decaimiento para Boltzmann |
| `tournament_m` | int | 5 | >= 2 | Tamano del torneo deterministico |
| `tournament_threshold` | float | 0.75 | [0.5, 1.0] | Umbral para torneo probabilistico |
| `non_uniform_b` | float | 1.0 | > 0 | Parametro b de decaimiento para mutacion no uniforme |
| `generational_gap` | float | 1.0 | [0, 1] | Brecha generacional G (fraccion de renovacion) |
| `max_seconds` | float | 0.0 | >= 0 | Tiempo maximo en segundos (0 = sin limite) |
| `content_threshold` | float | 0.0 | >= 0 | Umbral de cambio para criterio de contenido (0 = desactivado) |
| `content_generations` | int | 0 | >= 0 | Generaciones a evaluar para criterio de contenido (0 = desactivado) |
| `structure_threshold` | float | 0.0 | >= 0 | Umbral de diversidad para criterio de estructura (0 = desactivado) |
| `structure_generations` | int | 0 | >= 0 | Generaciones consecutivas para criterio de estructura (0 = desactivado) |
| `use_gpu` | bool | false | true/false | Usar renderer GPU con ModernGL |
| `gpu_device` | str | "auto" | "auto", "dedicated", "integrated" | Politica de seleccion GPU: default del sistema, dedicada obligatoria o integrada obligatoria |
| `min_error` | float | 0.0 | >= 0 | Error minimo para criterio de corte (0 = desactivado) |
| `gene_type` | str | "triangle" | "triangle", "ellipse" | Tipo de gen/figura geometrica |
| `arithmetic_alpha` | float | 0.5 | [0, 1] | Factor de interpolacion para crossover aritmetico |
| `gaussian_sigma` | float | 0.1 | > 0 | Desviacion estandar para mutacion gaussiana |
| `selection_methods` | list | [] | ver metodos abajo | Metodos de seleccion con pesos opcionales |
| `selection_weights` | list | [] | floats positivos | Pesos para seleccion ponderada de metodos de seleccion |
| `crossover_weights` | list | [] | floats positivos | Pesos para seleccion ponderada de crossover |
| `mutation_weights` | list | [] | floats positivos | Pesos para seleccion ponderada de mutacion |

En Linux hibrido, `gpu_device: "dedicated"` fuerza el intento de offload NVIDIA durante la creacion del contexto OpenGL. Si el contexto termina sobre otra GPU, la ejecucion falla en vez de continuar silenciosamente. `auto` respeta el contexto por defecto del sistema e `integrated` evita ese offload y valida que no haya quedado sobre una dedicada.

### Operadores disponibles

**Metodos de seleccion:** Elite, Ruleta, Universal, Ranking, Boltzmann, TorneosDeterministicos, TorneosProbabilisticos

**Metodos de crossover:** OnePoint, TwoPoint, Uniform, Annular, Aritmetico

**Metodos de mutacion:** Gen, MultiGen, Uniforme, Completa, NoUniforme, Gaussiana

**Estrategias de supervivencia:** Aditiva, Exclusiva

**Funciones de fitness:** MSE, MAE

### Operadores ponderados

Los campos de crossover, mutacion y seleccion soportan un formato con pesos para seleccion ponderada de operadores:

```json
{
    "crossover_methods": [
        {"method": "OnePoint", "weight": 3.0},
        {"method": "Uniform", "weight": 1.0}
    ]
}
```

Si se usan strings simples, todos los operadores tienen peso uniforme.

### Criterios de corte

El algoritmo se detiene cuando se cumple alguno de estos criterios:

1. **Generaciones maximas:** se alcanza `max_generations`
2. **Fitness alcanzado:** el mejor individuo supera `fitness_threshold`
3. **Tiempo maximo:** transcurren `max_seconds` segundos
4. **Error minimo:** el error del mejor individuo baja de `min_error`
5. **Contenido (estancamiento):** el fitness no mejora mas de `content_threshold` en `content_generations` generaciones
6. **Estructura (convergencia):** la desviacion estandar del fitness esta bajo `structure_threshold` por `structure_generations` generaciones consecutivas

## Archivos de salida

Todos los archivos se generan en la carpeta `output/`:

| Archivo | Descripcion |
|---|---|
| `output/final.png` | Mejor imagen generada al finalizar |
| `output/triangles.json` | Genes del mejor individuo serializados con metadatos |
| `output/metrics.csv` | Metricas por generacion: best_fitness, avg_fitness, fitness_std, elapsed_seconds |
| `output/gen_XXXX.png` | Snapshots intermedios cada `save_every` generaciones |

## Ejemplo de config.json

```json
{
    "triangle_count": 30,
    "population_size": 100,
    "max_generations": 500,
    "fitness_threshold": 0.95,
    "selection_method": "Boltzmann",
    "crossover_methods": ["Uniform"],
    "crossover_probability": 0.7,
    "mutation_methods": ["NoUniforme"],
    "mutation_rate": 0.1,
    "survival_strategy": "Aditiva",
    "fitness_function": "MSE",
    "k_offspring": 80,
    "save_every": 50,
    "boltzmann_t0": 100.0,
    "boltzmann_tc": 1.0,
    "boltzmann_k": 0.01,
    "generational_gap": 0.8,
    "max_seconds": 300,
    "content_threshold": 0.0001,
    "content_generations": 50
}
```

## Tests

Ejecutar la suite completa de tests:

```bash
python3 -m pytest tests/ -v
```

Sin tests de GPU (para entornos sin soporte):

```bash
python3 -m pytest tests/ --ignore=tests/test_gpu_renderer.py -v
```

Con reporte de tiempos:

```bash
python3 -m pytest tests/ -v --durations=10
```

## Experimentacion

El directorio `experiments/` contiene scripts para ejecutar baterias de experimentos sistematicos:

```bash
# Experimento de metodos de seleccion
python3 -m experiments.exp_seleccion

# Experimento de metodos de crossover
python3 -m experiments.exp_crossover

# Experimento de metodos de mutacion
python3 -m experiments.exp_mutacion

# Experimento de cantidad de triangulos
python3 -m experiments.exp_triangulos

# Experimento de probabilidad de crossover
python3 -m experiments.exp_pc

# Experimento de estrategias de supervivencia
python3 -m experiments.exp_supervivencia

# Experimento de criterios de corte
python3 -m experiments.exp_corte

# Experimento de operadores ponderados
python3 -m experiments.exp_ponderado
```

Los resultados se guardan en `experiments/results/` y los graficos generados en `experiments/plots/`.

Para generar graficos a partir de resultados existentes:

```bash
python3 -m experiments.plot_results
```

## Estructura del proyecto

```
config/          -- config loader y config.json
genes/           -- Gene ABC, TriangleGene, EllipseGene, PolygonGene
core/            -- Individual, Population, GeneticAlgorithm, MetricsCollector
selection/       -- estrategias de seleccion (Elite, Ruleta, Universal, etc.)
crossover/       -- operadores de crossover (OnePoint, TwoPoint, Uniform, etc.)
mutation/        -- operadores de mutacion (Gen, MultiGen, Uniforme, etc.)
fitness/         -- funciones de fitness (MSE, MAE)
survival/        -- estrategias de supervivencia (Aditiva, Exclusiva)
render/          -- renderers (CPU, GPU) y shaders
experiments/     -- scripts de experimentacion y graficos
output/          -- imagenes generadas, metrics.csv, triangles.json
tests/           -- tests unitarios e integracion
tasks/           -- especificaciones y trackeo de tareas SDD
```

## Autores

[Grupo 8] -- Sistemas de Inteligencia Artificial, ITBA
