# Plan de Presentacion TP2 - Algoritmos Geneticos para Aproximacion de Imagenes

**Duracion:** 25 minutos | **Equipo:** 4 speakers | **~62 slides**

**Nombres de archivos:** Todos los graficos llevan prefijo `<experimento>_<imagen>_` para identificarlos rapidamente.

---

## Como correr

```bash
# Todos los experimentos + plots (~1.5-2 horas)
bash experiments/run_all.sh

# Noche estrellada (separado, corre largo)
.venv/bin/python3 main.py images/noche.jpg run_configs/config_overnight_noche.json

# Solo plots de noche estrellada
.venv/bin/python experiments/plot_islas_noche.py
```

---

## BLOQUE 1: Introduccion y problema (~6 min) — Speaker 1

### Slide 1: Titulo
- Nombre del TP, equipo (Priotto, Lategana, Ahumada, Bassi), ITBA 2026

### Slide 2: El problema
- "Aproximar imagen con N triangulos semi-transparentes"
- Secuencia visual de la noche estrellada evolucionando

### Slide 3: Representacion del cromosoma
- Cada triangulo = 11 genes: `[x1, y1, x2, y2, x3, y3, R, G, B, alpha, active]`
- Diagrama visual, explicar que el orden = capas

### Slide 4: Complejidad de las imagenes
- `plots/complejidad/complejidad_fitness_best.png` — 4 curvas superpuestas
- `plots/complejidad/complejidad_image_grid.png` — grid con las 4 imagenes en distintas generaciones

### Slide 5: Triangulos - Ucrania
- `plots/num_triangulos/1_Ucrania/num_triangulos_1_Ucrania_fitness_best.png`
- `plots/num_triangulos/1_Ucrania/num_triangulos_1_Ucrania_image_grid.png`

### Slide 6: Triangulos - Grecia
- `plots/num_triangulos/2_Grecia/num_triangulos_2_Grecia_fitness_best.png`
- `plots/num_triangulos/2_Grecia/num_triangulos_2_Grecia_image_grid.png`

### Slide 7: Triangulos - Apple
- `plots/num_triangulos/3_Apple/num_triangulos_3_Apple_fitness_best.png`
- `plots/num_triangulos/3_Apple/num_triangulos_3_Apple_image_grid.png`

### Slide 8: Triangulos - Cubista
- `plots/num_triangulos/4_Cubista/num_triangulos_4_Cubista_fitness_best.png`
- `plots/num_triangulos/4_Cubista/num_triangulos_4_Cubista_image_grid.png`

### Slide 9: Triangulos - Resumen cruzado
- `plots/num_triangulos/num_triangulos_heatmap_fitness.png`
- `plots/num_triangulos/num_triangulos_barras_comparacion.png`
- **Punto:** Mas triangulos = mas detalle, rendimientos decrecientes

---

## BLOQUE 2: Operadores geneticos (~8 min) — Speaker 2

### Slides 10-13: Funciones de fitness (x4 imagenes)
- `plots/fitness/<IMG>/fitness_<IMG>_fitness_best.png`
- Compara: LinearMSE, MSE, MAE, SSIM

### Slide 14: Fitness - Resumen cruzado
- `plots/fitness/fitness_heatmap_fitness.png`
- `plots/fitness/fitness_heatmap_convergencia.png`
- **Punto:** LinearMSE mejor gradiente. SSIM se estanca (anecdota noche estrellada 0.66)

### Slides 15-18: Inicializacion (x4 imagenes)
- `plots/inicializacion/<IMG>/inicializacion_<IMG>_fitness_best.png`
- `plots/inicializacion/<IMG>/inicializacion_<IMG>_image_grid.png`
- Compara: Random vs SmartInit

### Slide 19: Inicializacion - Resumen cruzado
- `plots/inicializacion/inicializacion_heatmap_fitness.png`
- `plots/inicializacion/inicializacion_barras_comparacion.png`
- **Punto:** Smart init ventaja desde gen 0. Historia: color random -> centroide -> multi-escala

### Slides 20-23: Seleccion (x4 imagenes)
- `plots/seleccion/<IMG>/seleccion_<IMG>_fitness_best.png`
- Compara: Elite, Ruleta, Universal, Ranking, Boltzmann, TorneosDeterministicos, TorneosProbabilisticos

### Slide 24: Seleccion - Resumen cruzado
- `plots/seleccion/seleccion_heatmap_fitness.png`
- `plots/seleccion/seleccion_heatmap_convergencia.png`
- **Punto:** Boltzmann balancea exploracion/explotacion

### Slides 25-28: Crossover (x4 imagenes)
- `plots/crossover/<IMG>/crossover_<IMG>_fitness_best.png`
- Compara: OnePoint, TwoPoint, Uniform, Annular, Aritmetico

### Slide 29: Crossover - Resumen cruzado
- `plots/crossover/crossover_heatmap_fitness.png`
- `plots/crossover/crossover_barras_comparacion.png`
- **Punto:** Uniform mejor porque posicion en cromosoma no tiene significado semantico

### Slides 30-33: Mutacion (x4 imagenes)
- `plots/mutacion/<IMG>/mutacion_<IMG>_fitness_best.png`
- Compara: Gen, MultiGen, Uniforme, Completa, NoUniforme, Gaussiana

### Slide 34: Mutacion - Resumen cruzado
- `plots/mutacion/mutacion_heatmap_fitness.png`
- `plots/mutacion/mutacion_heatmap_convergencia.png`
- **Punto:** Gaussiana ajustes finos con decay. MultiGen explora mas.

### Slides 35-38: Supervivencia (x4 imagenes)
- `plots/supervivencia/<IMG>/supervivencia_<IMG>_fitness_best.png`
- Compara: Aditiva vs Exclusiva

### Slide 39: Supervivencia - Resumen cruzado
- `plots/supervivencia/supervivencia_heatmap_fitness.png`
- `plots/supervivencia/supervivencia_barras_comparacion.png`
- **Punto:** Aditiva conserva mejores. Exclusiva renueva pero arriesga.

---

## BLOQUE 3: Optimizaciones (~6 min) — Speaker 3

### Slides 40-43: Condiciones de corte (x4 imagenes)
- `plots/corte/<IMG>/corte_<IMG>_fitness_best.png`
- Compara: solo_gen, contenido_50, contenido_100, estructura_30, combinado

### Slide 44: Corte - Resumen cruzado
- `plots/corte/corte_heatmap_fitness.png`
- `plots/corte/corte_heatmap_convergencia.png`
- **Punto:** Combinado es el mas eficiente

### Slides 45-48: Anti-estancamiento (x4 imagenes)
- `plots/anti_estancamiento/<IMG>/anti_estancamiento_<IMG>_fitness_best.png`
- Compara: Con vs Sin anti-estancamiento

### Slide 49: Anti-estancamiento - Resumen cruzado
- `plots/anti_estancamiento/anti_estancamiento_heatmap_fitness.png`
- `plots/anti_estancamiento/anti_estancamiento_barras_comparacion.png`
- **Punto:** Mutation boost x4 + reinicio 20% rompe mesetas

### Slides 50-53: GPU (x4 imagenes)
- `plots/gpu/<IMG>/gpu_<IMG>_benchmark.png`
- Barras CPU vs GPU + speedup

### Slides 54-57: Formas (x4 imagenes)
- `plots/formas/<IMG>/formas_<IMG>_fitness_best.png`
- `plots/formas/<IMG>/formas_<IMG>_image_grid.png`
- Compara: Triangulo vs Elipse

### Slide 58: Formas - Resumen cruzado
- `plots/formas/formas_heatmap_fitness.png`
- `plots/formas/formas_barras_comparacion.png`
- **Punto:** Elipses cubren mas area. Triangulos capturan bordes rectos.

---

## BLOQUE 4: Gran final (~5 min) — Speaker 4

### Slide 59: Modelo de islas - Explicacion
- Diagrama topologia anillo, 6 islas, migracion cada 30 gen

### Slide 60: Islas - Configuracion
- Tabla:

| Isla | Mutacion | Sigma | Seleccion | Estrategia |
|------|----------|-------|-----------|------------|
| Exploradora agresiva | 20% | 0.25 | Ruleta | Explorar radical |
| Exploradora suave | 12% | 0.15 | Universal | Exploracion moderada |
| Conservadora elite | 2% | 0.04 | Elite | Refinar mejores |
| Torneo agresivo | 10% | - | Torneo m=8 | Presion selectiva |
| Boltzmann lenta | 6% | - | Boltzmann t0=300 | Convergencia gradual |
| Refinadora ranking | 3% | 0.06 | Ranking | Fine-tuning |

### Slide 61: Noche Estrellada - Resultados
- `plots/islas_noche/islas_noche_fitness.png` — curvas 6 islas
- `plots/islas_noche/islas_noche_snapshots.png` — grid gen 0→3000
- `plots/islas_noche/islas_noche_diversidad.png` — diversidad por isla

### Slide 62: Conclusion
- Mejor combo: LinearMSE + Gaussiana + Uniform + Boltzmann + SmartInit + Islas
- Lecciones: SSIM no sirve como fitness, smart init importa, GPU es clave
- Mejoras futuras

---

## Nomenclatura de archivos

Todos los graficos siguen el patron:
```
experiments/plots/<experimento>/<imagen>/<experimento>_<imagen>_<tipo>.png
```

Ejemplos:
- `seleccion/1_Ucrania/seleccion_1_Ucrania_fitness_best.png`
- `mutacion/3_Apple/mutacion_3_Apple_image_grid.png`
- `crossover/crossover_heatmap_fitness.png` (resumen cruzado, sin imagen)
- `gpu/2_Grecia/gpu_2_Grecia_benchmark.png`
- `islas_noche/islas_noche_fitness.png`

Con este patron, buscando por nombre sabes exactamente que experimento, que imagen, y que tipo de grafico es.
