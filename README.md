# Prácticas de Visión por Computador

Repositorio con las prácticas de la asignatura **Visión por Computador** del Grado en **Ingeniería Informática** de la **Universidad de Las Palmas de Gran Canaria (ULPGC)**, curso 2026/27.

Cada práctica tiene su propia carpeta con un `README.md` que explica el trabajo realizado, los resultados y el uso de herramientas de IA.

## Prácticas

| Práctica | Tema | Notebooks | Memoria |
|---|---|---|---|
| [Práctica 1](PRACTICA_1) | Primeros pasos con OpenCV: tablero de ajedrez, Mondrian, webcam y pop art | [`VC_P1.ipynb`](PRACTICA_1/VC_P1.ipynb) | [README](PRACTICA_1/README.md) |
| [Práctica 2](PRACTICA_2) | Detección de bordes (Canny y Sobel) y demostrador de privacidad con sustracción de fondo | [`VC_P2_TAREAS.ipynb`](PRACTICA_2/VC_P2_TAREAS.ipynb) | [README](PRACTICA_2/README.md) |
| [Práctica 3](PRACTICA_3) | Detección y reconocimiento de formas: conteo de monedas y clasificación de microplásticos | [`Tarea 1`](PRACTICA_3/TAREAS/VC_P3_TAREA_1.ipynb), [`Tarea 2`](PRACTICA_3/TAREAS/VC_P3_TAREA_2.ipynb) | [README](PRACTICA_3/README.md) |
| [Práctica 4](PRACTICA_4) | Detección y seguimiento de vehículos y personas, detección de matrículas con YOLO y lectura con OCR | [`Tarea 1`](PRACTICA_4/TAREAS/VC_P4_TAREA_1.ipynb), [`Tarea 2`](PRACTICA_4/TAREAS/VC_P4_TAREA_2.ipynb) | *En desarrollo* |

## Estructura

Las prácticas siguen la misma organización (cada una incluye solo las carpetas que necesita):

```
PRACTICA_N/
├── README.md          Memoria de la práctica
├── VC_PN.ipynb        Notebook de ejemplos de la práctica
├── TAREAS/            Notebooks con la resolución de las tareas
├── data/              Datos de entrada (imágenes, anotaciones...)
├── img/               Imágenes de entrada pequeñas o usadas en el README
└── videos/            Vídeos de entrada
```

La Práctica 4 añade además:

```
PRACTICA_4/
├── ENUNCIADO-P4.md    Enunciado de la parte de detección
├── ENUNCIADO-P4b.md   Enunciado de la parte de OCR
├── ejemplos/          Notebooks de ejemplo (YOLO, RT-DETR, OCR)
├── modelos/           Modelos preentrenados de YOLO
├── dataset/           Dataset de matrículas (en el repo solo el script, data.yaml y ground truth)
└── docs/              Material de apoyo
```

## Tecnologías

- Python 3
- [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/) y [Pillow](https://python-pillow.org/)
- [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) y [seaborn](https://seaborn.pydata.org/) para métricas y matrices de confusión (Práctica 3)
- [Ultralytics YOLO](https://docs.ultralytics.com/) para detección, seguimiento y entrenamiento (Práctica 4)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR), [Tesseract](https://github.com/tesseract-ocr/tesseract) y [Transformers](https://huggingface.co/docs/transformers) para OCR (Práctica 4)
- Jupyter Notebook

## Instalación

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
```

### Prácticas 1 a 3

Entorno `VC_P1` con Conda. En Windows se puede recrear exactamente con la lista de paquetes incluida:

```bash
conda create --name VC_P1 --file spec-list.txt
conda activate VC_P1
```

O bien instalando las dependencias con pip:

```bash
pip install opencv-python numpy matplotlib pillow jupyter pandas scikit-learn seaborn
```

### Práctica 4

Entorno `VC_P4` (Python 3.9.5, como indica el enunciado):

```bash
conda create --name VC_P4 python=3.9.5
conda activate VC_P4
pip install ultralytics lap easyocr pytesseract transformers
```

Para entrenar con GPU hay que instalar PyTorch con CUDA siguiendo las [instrucciones oficiales](https://pytorch.org/get-started/locally/). Las imágenes del dataset de matrículas no se incluyen en el repositorio por su tamaño (~7 GB). Una vez descargadas en `PRACTICA_4/dataset/`, el script [`preparar_dataset.py`](PRACTICA_4/dataset/preparar_dataset.py) genera el reparto train/val/test, las etiquetas YOLO, el `data.yaml` y el ground truth.

> Los demostradores en vivo necesitan una webcam conectada.

## Autores

- **Pablo**, estudiante de Ingeniería Informática en la ULPGC.
- **Roberto**, estudiante de Ingeniería Informática en la ULPGC.
