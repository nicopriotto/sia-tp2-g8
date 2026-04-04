import sys
from PIL import Image
import numpy as np
from config.config_loader import load_config


def main():
    if len(sys.argv) < 3:
        print("Uso: python main.py <image_path> <config_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    config_path = sys.argv[2]

    config = load_config(config_path)

    # Cargar imagen objetivo como RGBA float32 [0,1]
    img = Image.open(image_path).convert("RGBA")
    target = np.array(img, dtype=np.float32) / 255.0

    print(f"Imagen: {image_path} ({target.shape[1]}x{target.shape[0]})")
    print(f"Triangulos: {config.triangle_count}")
    print(f"Poblacion: {config.population_size}")
    print(f"Generaciones max: {config.max_generations}")
    print(f"Seleccion: {config.selection_method}")
    print(f"Cruza: {config.crossover_methods}")
    print(f"Mutacion: {config.mutation_methods}")
    print(f"Fitness: {config.fitness_function}")


if __name__ == "__main__":
    main()
