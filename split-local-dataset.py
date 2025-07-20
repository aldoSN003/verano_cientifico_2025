import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def create_dataset_split(source_dir, output_dir, test_ratio=0.3, random_seed=42):
    """
    Crear división de entrenamiento/prueba para dataset de imágenes con distribución balanceada por clase.

    Args:
        source_dir: Ruta al directorio que contiene las carpetas de clases
        output_dir: Ruta donde se crearán las carpetas train/test
        test_ratio: Proporción de datos para prueba (por defecto: 0.2 = 20%)
        random_seed: Semilla aleatoria para reproducibilidad
    """

    # Establecer semilla aleatoria para reproducibilidad
    random.seed(random_seed)

    # Crear directorios de salida
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"

    # Eliminar directorios de salida existentes si existen
    if train_dir.exists():
        shutil.rmtree(train_dir)
    if test_dir.exists():
        shutil.rmtree(test_dir)

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Obtener todas las carpetas de clases
    source_path = Path(source_dir)
    class_folders = [d for d in source_path.iterdir() if d.is_dir()]

    split_info = defaultdict(lambda: {'train': 0, 'test': 0})

    print(f"Creando división train/test con {test_ratio*100}% para prueba...")
    print("-" * 50)

    for class_folder in class_folders:
        class_name = class_folder.name

        # Crear directorios de clase en carpetas train y test
        (train_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)

        # Obtener todos los archivos de imagen en esta carpeta de clase
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in class_folder.iterdir()
                      if f.is_file() and f.suffix.lower() in image_extensions]

        # Mezclar los archivos
        random.shuffle(image_files)

        # Calcular la división
        total_images = len(image_files)
        test_count = int(total_images * test_ratio)
        train_count = total_images - test_count

        # Dividir archivos
        test_files = image_files[:test_count]
        train_files = image_files[test_count:]

        # Copiar archivos a los directorios respectivos
        for file in train_files:
            shutil.copy2(file, train_dir / class_name / file.name)

        for file in test_files:
            shutil.copy2(file, test_dir / class_name / file.name)

        split_info[class_name]['train'] = train_count
        split_info[class_name]['test'] = test_count

        print(f"{class_name}: {total_images} total → {train_count} entrenamiento, {test_count} prueba")

    # Imprimir resumen
    print("-" * 50)
    total_train = sum(info['train'] for info in split_info.values())
    total_test = sum(info['test'] for info in split_info.values())
    total_all = total_train + total_test

    print(f"Resumen:")
    print(f"  Total de imágenes: {total_all}")
    print(f"  Imágenes de entrenamiento: {total_train} ({total_train/total_all*100:.1f}%)")
    print(f"  Imágenes de prueba: {total_test} ({total_test/total_all*100:.1f}%)")
    print(f"\nDivisión del dataset creada exitosamente")
    print(f"  Carpeta de entrenamiento: {train_dir}")
    print(f"  Carpeta de prueba: {test_dir}")

    return split_info

# Ejemplo de uso con tu dataset
if __name__ == "__main__":

    SOURCE_DIRECTORY = "./original-dataset"  # Directorio que contiene las carpetas de clases
    OUTPUT_DIRECTORY = "./lemon-leaf-disease-dataset"  # Donde se crearán las carpetas train/test

    # Crear la división
    split_results = create_dataset_split(
        source_dir=SOURCE_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        test_ratio=0.3,  # 30% para prueba, 70% para entrenamiento
        random_seed=42   # Para resultados reproducibles
    )


    print("\nDesglose detallado:")
    for class_name, counts in split_results.items():
        total = counts['train'] + counts['test']
        train_pct = counts['train'] / total * 100
        test_pct = counts['test'] / total * 100
        print(f"  {class_name}: {counts['train']} entrenamiento ({train_pct:.1f}%), "
              f"{counts['test']} prueba ({test_pct:.1f}%)")