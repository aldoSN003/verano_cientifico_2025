# Dataset de Clasificación de Enfermedades de Plantas

## Descripción del Dataset

Este dataset contiene imágenes de hojas de plantas con varias enfermedades y hojas saludables para tareas de clasificación.

### Resumen del Dataset

- **Total de Imágenes**: 1354
- **Imágenes de Entrenamiento**: 952
- **Imágenes de Prueba**: 402
- **Clases**: 9

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

Anthracnose, Bacterial Blight, Citrus Canker, Curl Virus, Deficiency Leaf, Dry Leaf, Healthy Leaf, Sooty Mould, Spider Mites

## Estructura del Dataset

```python
from datasets import load_dataset

# Cargar dataset
dataset = load_dataset("AldoSN/lemon-leaf-disease-dataset")

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




