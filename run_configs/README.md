# Run Configs

Configs JSON de ejemplo para ejecutar el algoritmo sin editar manualmente una configuracion desde cero.

## Uso

```bash
python3 main.py <ruta_imagen> <ruta_config_json>
```

Ejemplos:

```bash
python3 main.py images/1.jpg run_configs/base-config.json
python3 main.py images/3.jpg run_configs/config.json
```

## Configs disponibles

- `base-config.json`: base general con todos los campos.
- `config.json`: config de uso general.
- `bajo.json`: configuracion orientada a imagenes simples.
- `medio.json`: configuracion intermedia.
- `alto.json`: configuracion para imagenes complejas.
- `config_overnight_noche.json`: corrida larga sobre noche estrellada.
- `config_islands.json`: ejemplo de Island Model.
- `config_islands_diverse.json`: Island Model con islas mas diversas.

## Output esperado

Cada corrida genera en `output/`:

- `final.png`
- `triangles.json`
- `metrics.csv`
- `gen_XXXX.png` (si la config guarda snapshots)
