import os
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import HfApi, login
from pathlib import Path
import json
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Paso 1: Instalar paquetes requeridos (ejecutar en terminal)
"""
pip install datasets huggingface_hub pillow python-dotenv
"""


# Paso 2: Iniciar sesión en Hugging Face
def setup_hf_login():

    hf_token = os.getenv("HUGGING_FACE_TOKEN")

    if not hf_token:
        raise ValueError("HUGGING_FACE_TOKEN no encontrado en archivo .env. Por favor agrega tu token de Hugging Face.")

    try:
        login(token=hf_token)
        print("Inicio de sesión exitoso en Hugging Face")
    except Exception as e:
        raise ValueError(f"Error al iniciar sesión en Hugging Face: {e}")


# Paso 3: Crear dataset desde tus carpetas divididas
def create_hf_dataset(dataset_path, dataset_name="plant-disease-lemon-dataset"):
    """
    Crear dataset de Hugging Face desde carpetas divididas train/test

    Args:
        dataset_path: Ruta a tu carpeta de dataset que contiene directorios train/test
        dataset_name: Nombre para tu dataset en Hugging Face
    """

    dataset_path = Path(dataset_path)

    # Etiquetas de clase
    class_labels = [
        "Anthracnose", "Bacterial Blight", "Citrus Canker", "Curl Virus",
        "Deficiency Leaf", "Dry Leaf", "Healthy Leaf", "Sooty Mould", "Spider Mites"
    ]

    # Crear mapeo de etiqueta a índice
    label_to_id = {label: idx for idx, label in enumerate(class_labels)}

    def load_split_data(split_path):
        """Cargar imágenes y etiquetas desde un directorio de división"""
        images = []
        labels = []

        for class_folder in split_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                label_id = label_to_id[class_name]

                # Obtener todos los archivos de imagen
                image_extensions = {'.jpg', '.jpeg', '.png'}
                image_files = [f for f in class_folder.iterdir()
                               if f.is_file() and f.suffix.lower() in image_extensions]

                for img_file in image_files:
                    images.append(str(img_file))
                    labels.append(label_id)

        return images, labels

    # Cargar datos de entrenamiento y prueba
    train_images, train_labels = load_split_data(dataset_path / "train")
    test_images, test_labels = load_split_data(dataset_path / "test")

    # Crear datasets
    train_dataset = Dataset.from_dict({
        "image": train_images,
        "label": train_labels
    }).cast_column("image", Image())

    test_dataset = Dataset.from_dict({
        "image": test_images,
        "label": test_labels
    }).cast_column("image", Image())

    # Crear DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Agregar información de características
    dataset_dict = dataset_dict.cast_column("image", Image())

    print(f"¡Dataset creado exitosamente!")
    print(f"Muestras de entrenamiento: {len(train_dataset)}")
    print(f"Muestras de prueba: {len(test_dataset)}")
    print(f"Clases: {class_labels}")

    return dataset_dict, class_labels


# Paso 4: Subir a Hugging Face Hub
def upload_to_hf(dataset_dict, class_labels, repo_name, username=None, private=False):
    """

    Args:
        dataset_dict: Objeto DatasetDict
        class_labels: Lista de etiquetas de clase
        repo_name: Nombre del repositorio en Hugging Face
        username: Nombre de usuario de Hugging Face (opcional)
        private: Variable booleana para determinar el acceso publico al repositorio
    """

    # Crear nombre del repositorio
    if username:
        full_repo_name = f"{username}/{repo_name}"
    else:
        full_repo_name = repo_name

    print(f"Subiendo a: {full_repo_name}")

    # Empujar al hub
    dataset_dict.push_to_hub(
        full_repo_name,
        private=private,
        commit_message="Subida inicial del dataset de clasificación de enfermedades de limon"
    )

    print(f"¡Dataset subido exitosamente!")
    print(f"URL: https://huggingface.co/datasets/{full_repo_name}")

    return full_repo_name


# Paso 5: Crear tarjeta del dataset (README)
def create_dataset_card(repo_name, class_labels, train_count, test_count):
    """
    Crear una tarjeta comprensiva del dataset
    """

    dataset_card = f"""# Dataset de Clasificación de Enfermedades de Plantas

## Descripción del Dataset

Este dataset contiene imágenes de hojas de plantas con varias enfermedades y hojas saludables para tareas de clasificación.

### Resumen del Dataset

- **Total de Imágenes**: {train_count + test_count}
- **Imágenes de Entrenamiento**: {train_count}
- **Imágenes de Prueba**: {test_count}
- **Clases**: {len(class_labels)}

### Distribución de Clases

| Clase | Entrenamiento | Prueba | Total |
|-------|---------------|--------|-------|
| Anthracnose | 70 | 30 | 100 |
| Bacterial Blight | 74 | 31 | 105 |
| Citrus Canker | 125 | 53 | 178 |
| Curl Virus | 81 | 34 | 115 |
| Deficiency Leaf | 136 | 57 | 193 |
| Dry Leaf | 131 | 55 | 186 |
| Healthy Leaf | 147 | 63 | 210 |
| Sooty Mould | 108 | 45 | 153 |
| Spider Mites | 80 | 34 | 114 |

### Clases

{', '.join(class_labels)}

## Estructura del Dataset

```python
from datasets import load_dataset

# Cargar dataset
dataset = load_dataset("{repo_name}")

# Acceder a las divisiones train/test
train_data = dataset["train"]
test_data = dataset["test"]

# Ejemplo de uso
for example in train_data:
    image = example["image"]
    label = example["label"]
    # Procesar tus datos
```

## Uso

Este dataset es adecuado para:
- Clasificación de enfermedades de plantas
- Investigación en visión por computadora
- Aplicaciones de IA agrícola
- Propósitos educativos




"""

    return dataset_card


# Paso 6: Proceso completo de subida
def complete_upload_process(dataset_path, repo_name, username=None, private=False):
    """
    Proceso completo para subir dataset a Hugging Face
    """

    print("Paso 1: Configurando inicio de sesión en Hugging Face...")
    setup_hf_login()

    print("Paso 2: Creando dataset...")
    dataset_dict, class_labels = create_hf_dataset(dataset_path)

    print("Paso 3: Subiendo a Hugging Face Hub...")
    full_repo_name = upload_to_hf(dataset_dict, class_labels, repo_name, username, private)

    print("Paso 4: Creando tarjeta del dataset...")
    train_count = len(dataset_dict["train"])
    test_count = len(dataset_dict["test"])
    dataset_card = create_dataset_card(full_repo_name, class_labels, train_count, test_count)

    # Guardar tarjeta del dataset localmente
    with open("README.md", "w") as f:
        f.write(dataset_card)

    print("Tarjeta del dataset guardada como README.md")

    return full_repo_name, dataset_card


if __name__ == "__main__":
    # Configuración
    DATASET_PATH = "./lemon-leaf-disease-dataset"
    REPO_NAME = "lemon-leaf-disease-dataset"
    USERNAME = "AldoSN"
    PRIVATE = False


    repo_name, card = complete_upload_process(
        dataset_path=DATASET_PATH,
        repo_name=REPO_NAME,
        username=USERNAME,
        private=PRIVATE
    )

    print(f"Subida completa")
    print(f"URL del Dataset: https://huggingface.co/datasets/{repo_name}")
