# Decisiones de Diseno — TP2 Algoritmos Geneticos

> Documento de referencia canonica para todas las decisiones de diseño del algoritmo genetico
> que aproxima imagenes usando triangulos semi-transparentes.

---

## Preguntas de Diseno

### 1. Como evaluo mi aproximacion al dibujo?

Comparacion pixel a pixel entre la imagen generada y la imagen objetivo. La metrica principal es **MSE (Mean Squared Error)** sobre los canales RGB de ambas imagenes normalizadas a [0, 1].

La funcion de fitness transforma el error en un valor a maximizar:

```
fitness = 1 / (1 + MSE)
```

- Rango resultante: (0, 1], donde 1.0 significa imagenes identicas.
- Mayor fitness = mejor aproximacion.

Se elige MSE sobre MAE porque penaliza mas los errores grandes: un pixel muy diferente es perceptualmente peor que muchos pixeles ligeramente diferentes. Esto genera un gradiente mas fuerte hacia soluciones que eliminan los errores mas notorios primero.

### 2. Que es un individuo?

Un individuo es una lista ordenada de N triangulos (genes). Cada gen (triangulo) se define como una tupla de 10 atributos:

```
(x1, y1, x2, y2, x3, y3, r, g, b, a)
```

- **Coordenadas espaciales** `(x1, y1, x2, y2, x3, y3)`: valores float normalizados en [0, 1]. Se mapean a pixeles multiplicando por el ancho/alto de la imagen.
- **Color RGB** `(r, g, b)`: enteros en [0, 255].
- **Alpha** `(a)`: float en [0, 1] representando la opacidad del triangulo.

El orden de los genes importa porque los triangulos se renderizan secuencialmente: los posteriores se dibujan encima de los anteriores (alpha compositing).

### 3. Que es el fitness?

La similitud entre la imagen generada al renderizar el individuo y la imagen objetivo. Se calcula como:

```
fitness = 1 / (1 + MSE)
```

donde MSE se computa sobre los canales RGB de ambas imagenes normalizadas a [0, 1]. Mayor fitness = mejor aproximacion. Un fitness de 1.0 indica imagenes identicas.

### 4. Como mutar?

Dos estrategias de mutacion sobre un gen (alelo), definidas en las slides:

**(a) Reemplazo completo (Gen):** Se genera un gen completamente nuevo con valores aleatorios. Util para explorar regiones distantes del espacio de busqueda.

**(b) Delta incremental (Aditiva):** Se suman pequenos deltas a los atributos del gen existente, con clampeo a los rangos validos. Util para refinamiento local. El delta es proporcional a un parametro `strength`: `nuevo = actual + Uniform(-strength * rango, strength * rango)`.

La mutacion se aplica gen por gen con una probabilidad configurable (`mutation_rate` o Pm).

Si la mutacion delta produce un triangulo degenerado (area = 0, vertices colineales), se reintenta con `strength * 2` hasta 10 veces. Si todos fallan, se devuelve el gen original sin mutar. Esto preserva la localidad de la mutacion sin introducir ruido aleatorio.

### 5. Como cruzar?

- **One Point:** Se elige un punto de corte aleatorio y se intercambian los genes a partir de ese punto. Preserva la correlacion posicional entre genes adyacentes.
- **Two Point:** Se eligen dos puntos de corte y se intercambia el segmento entre ellos. Tambien preserva correlacion posicional.
- **Uniform:** Cada gen se toma de uno u otro padre con probabilidad 0.5. No preserva correlacion posicional, pero puede ser mejor cuando no existe correlacion real entre posiciones.

Dado que el orden de los triangulos afecta el renderizado (superposicion), One Point y Two Point pueden preservar sub-secuencias utiles de capas. La cruza puede generar buenos descendientes porque un hijo hereda zonas bien aproximadas de cada padre: si padre A tiene bien el cielo y padre B tiene bien el suelo, el hijo puede tener ambas cosas bien.

La probabilidad de crossover (Pc) controla si la cruza efectivamente ocurre. Su rol y uso concreto se detallan en la seccion siguiente.

### Probabilidad de Recombinacion (Pc)

**Definicion de Pc**

- Pc (`crossover_probability`) es la probabilidad de que el operador de crossover se aplique al aparear dos padres.
- Con probabilidad Pc: se ejecuta el crossover y se generan hijos recombinados.
- Con probabilidad `1 - Pc`: los hijos son copias exactas de los padres, sin recombinacion.
- En ambos casos, los hijos pasan por la etapa de mutacion posterior.

En este proyecto, este parametro ya existe en la configuracion como `crossover_probability` dentro de la dataclass `Config`.

**Por que existe Pc**

- Protege combinaciones de genes que funcionan bien juntas. Si un individuo ya tiene un conjunto de triangulos en posiciones utiles, el crossover puede romper esa combinacion.
- Permite que buenos genotipos pasen a la siguiente generacion con solo mutaciones menores, sin forzar recombinacion en todos los apareamientos.
- Agrega control sobre el balance entre exploracion y explotacion del algoritmo.

**Valores tipicos y extremos**

- Rango tipico: Pc entre 0.6 y 0.9.
- `Pc = 1.0`: el crossover siempre ocurre, sin proteccion de combinaciones utiles.
- `Pc = 0.0`: el crossover nunca ocurre y la unica fuente de variacion pasa a ser la mutacion.

**Donde se aplica en el loop del GA**

```python
if random() < crossover_probability:
    hijo1, hijo2 = crossover_op.crossover(p1, p2)
else:
    hijo1, hijo2 = p1.copy(), p2.copy()

# En ambos casos, aplicar mutacion despues
```

Este pseudocodigo se implementara de forma concreta en el loop principal del algoritmo genetico.

### 6. Version mas simple posible?

- 1 triangulo por individuo.
- 5 individuos en la poblacion.
- Seleccion: Elite.
- Cruza: One Point (trivial con 1 gen, pero funcional).
- Mutacion: Gen (reemplazo completo).
- 50 generaciones.
- Imagen objetivo: cuadrado de color solido.

Esto permite validar el pipeline completo end-to-end antes de agregar complejidad.

### 7. Que imagenes y cuantos triangulos?

- Empezar con imagenes simples: banderas (rectangulares, colores planos), emojis (formas geometricas basicas).
- 10-20 triangulos para validacion inicial.
- Escalar a 50-200 triangulos para imagenes mas complejas una vez validado el sistema.
- Mas triangulos = cromosoma mas largo = mas tiempo por evaluacion de fitness y mas espacio de busqueda.

### 8. Se puede implementar parcialmente?

Si. Con solo Elite + One Point + Gen + Aditiva ya se puede evaluar si el sistema converge (el fitness sube con las generaciones). Los demas metodos de seleccion, cruza y mutacion se agregan de forma incremental.

---

## Representacion del Genotipo

### Atributos del gen (TriangleGene)

| Atributo | Tipo | Rango | Descripcion |
|---|---|---|---|
| `x1` | float | [0, 1] | Coordenada X del vertice 1 (normalizada) |
| `y1` | float | [0, 1] | Coordenada Y del vertice 1 (normalizada) |
| `x2` | float | [0, 1] | Coordenada X del vertice 2 (normalizada) |
| `y2` | float | [0, 1] | Coordenada Y del vertice 2 (normalizada) |
| `x3` | float | [0, 1] | Coordenada X del vertice 3 (normalizada) |
| `y3` | float | [0, 1] | Coordenada Y del vertice 3 (normalizada) |
| `r` | int | [0, 255] | Componente rojo del color |
| `g` | int | [0, 255] | Componente verde del color |
| `b` | int | [0, 255] | Componente azul del color |
| `a` | float | [0, 1] | Opacidad (0.0 = transparente, 1.0 = opaco) |

### Coordenadas normalizadas

Las coordenadas espaciales estan normalizadas en [0, 1] para ser independientes del tamano de la imagen. El mapeo a pixeles ocurre en el renderer:

```
pixel_x = int(x * width)
pixel_y = int(y * height)
```

Esto permite que el mismo genotipo se renderice a cualquier resolucion sin modificar los genes.

### Alpha y color

- **Alpha** en [0, 1]: 0.0 = completamente transparente (el triangulo no aporta nada), 1.0 = completamente opaco (cubre totalmente lo que hay debajo).
- **RGB** en [0, 255]: formato entero estandar. Se convierte a RGBA de Pillow como `(r, g, b, int(a * 255))` para el renderizado.

### Genotipo y fenotipo

- **Genotipo**: lista ordenada de N genes (`list[TriangleGene]`). El individuo completo es esta lista mas su valor de fitness.
- **Fenotipo**: imagen RGBA resultante de renderizar los triangulos en orden sobre un canvas blanco (255, 255, 255, 255). Cada triangulo se compone sobre el canvas usando alpha compositing (Porter-Duff "source over").

---

## Justificacion de la Funcion de Aptitud

### Por que MSE

El Mean Squared Error penaliza los errores cuadraticamente, lo cual es deseable para aproximacion de imagenes:

- Un pixel muy diferente es perceptualmente peor que muchos pixeles ligeramente diferentes.
- El gradiente cuadratico genera mayor presion selectiva hacia la eliminacion de errores grandes primero.
- Es una metrica estandar en procesamiento de imagenes con buenas propiedades matematicas.

### Formula completa

```
MSE = (1 / N) * sum((gen_i - target_i)^2)
```

donde la suma recorre todos los pixeles y los 3 canales RGB (se ignora el canal alpha). N es el numero total de valores comparados (height * width * 3).

El MSE se calcula solo sobre RGB porque el canal alpha de la imagen generada es un artefacto del proceso de compositing, no una propiedad visual comparable con el alpha de la imagen objetivo.

### Transformacion a fitness

```
fitness = 1 / (1 + MSE)
```

Esta transformacion mapea MSE en [0, inf) a fitness en (0, 1]:

| MSE | Fitness | Interpretacion |
|---|---|---|
| 0.0 | 1.0 | Imagenes identicas |
| 0.5 | 0.667 | Diferencia moderada |
| 1.0 | 0.5 | Maxima diferencia (blanco vs negro) |

Es monotona decreciente en MSE, lo cual garantiza que menor error = mayor fitness.

### Comparacion con MAE

```
MAE = (1 / N) * sum(|gen_i - target_i|)
```

- **MAE** es mas robusto a outliers: un pixel muy diferente no domina tanto el error total.
- **MSE** penaliza mas los errores grandes, generando mayor gradiente para corregirlos.
- Para este problema, MSE es preferible porque queremos eliminar los defectos visuales mas notorios primero. MAE se implementa como alternativa (TASK-029) para experimentacion comparativa.

---

## Brecha Generacional (G)

G controla que fraccion de la poblacion se renueva por generacion.

**Formula:** `k = max(2, int(G * N))`, ajustado a par si es impar.

**Interaccion con estrategias de supervivencia:**
- Con **Aditiva**: se forma un pool de (N + k) individuos, se seleccionan N.
- Con **Exclusiva**: si k >= N, solo hijos; si k < N, todos los hijos + (N-k) padres.

| G | N=20 | k_offspring | Comportamiento |
|---|------|-------------|----------------|
| 1.0 | 20 | 20 | Reemplazo completo |
| 0.5 | 20 | 10 | Mitad se renueva |
| 0.1 | 20 | 2 | Minimo: solo 2 hijos |
| 0.0 | 20 | 2 | Minimo forzado |
