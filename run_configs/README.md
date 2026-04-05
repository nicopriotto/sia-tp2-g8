# Run Configs

Configs de ejemplo para probar el algoritmo genetico sin tener que editar `config.json`.

## Como correrlos

Activar el entorno virtual:

```bash
source .venv/bin/activate
```

Ejecutar el programa indicando una imagen objetivo y uno de estos configs:

```bash
python main.py /ruta/a/tu/imagen.png run_configs/quick-smoke-20gen.json
python main.py /ruta/a/tu/imagen.png run_configs/baseline-100gen.json
```

## Configs disponibles

- `quick-smoke-20gen.json`: corrida corta para verificar que todo funciona rapido.
- `baseline-100gen.json`: corrida base un poco mas larga para observar mejor la evolucion.

## Output

Cada corrida genera:

- `output/final.png`
- `output/triangles.json`
