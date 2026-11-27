# Prácticas de Visión por Computador

Repositorio con las prácticas de la asignatura **Visión por Computador** del Grado en **Ingeniería Informática** de la **Universidad de Las Palmas de Gran Canaria (ULPGC)**, curso 2026/27.

Cada práctica tiene su propia carpeta con un `README.md` que explica el trabajo realizado, los resultados y el uso de herramientas de IA.

## Prácticas

| Práctica | Tema | Notebooks | Memoria |
|---|---|---|---|
| [Práctica 1](PRACTICA_1) | Primeros pasos con OpenCV: tablero de ajedrez, Mondrian, webcam y pop art | [`VC_P1.ipynb`](PRACTICA_1/VC_P1.ipynb) | [README](PRACTICA_1/README.md) |
| [Práctica 2](PRACTICA_2) | Detección de bordes (Canny y Sobel) y demostrador de privacidad con sustracción de fondo | [`VC_P2_TAREAS.ipynb`](PRACTICA_2/VC_P2_TAREAS.ipynb) | [README](PRACTICA_2/README.md) |
| [Práctica 3](PRACTICA_3) | Detección y reconocimiento de formas: conteo de monedas y clasificación de microplásticos | [`Tarea 1`](PRACTICA_3/TAREAS/VC_P3_TAREA_1.ipynb), [`Tarea 2`](PRACTICA_3/TAREAS/VC_P3_TAREA_2.ipynb) | [README](PRACTICA_3/README.md) |
| [Práctica 4](PRACTICA_4) | Detección y seguimiento de vehículos y personas, detección de matrículas con YOLO y lectura con OCR (EasyOCR vs SmolVLM) | [`Tarea 1`](PRACTICA_4/TAREAS/VC_P4_TAREA_1.ipynb), [`Tarea 2`](PRACTICA_4/TAREAS/VC_P4_TAREA_2.ipynb) | [README](PRACTICA_4/README.md) |
| [Práctica 5](PRACTICA_5) | Biometría facial: espejo de emociones con un modelo propio (FER-2013, HOG+SVM vs MobileNetV2) y filtro Super Saiyan con landmarks de cara y manos | [`Tarea 1`](PRACTICA_5/TAREAS/VC_P5_TAREA_1.ipynb), [`Tarea 2`](PRACTICA_5/TAREAS/VC_P5_TAREA_2.ipynb) | [README](PRACTICA_5/README.md) |

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
├── TAREAS/runs/       Entrenamiento del detector de matrículas (gráficas y best.pt)
└── resultados/        CSV, gráficas y fotogramas de resultados (los vídeos no se suben por tamaño)
```

La Práctica 5 sigue la misma idea:

```
PRACTICA_5/
├── ENUNCIADO-P5.md    Enunciado de la práctica
├── ejemplos/          Notebooks de ejemplo (detectores, MediaPipe, autocaras, LBP/HOG, deepface)
├── modelos/           Detector DNN, modelos de MediaPipe y modelos de emociones entrenados
├── data/              Datasets de caras (no se suben: se descargan aparte)
├── img/recursos/      Imágenes PNG de los filtros (skins) de los prototipos
└── resultados/        Gráficas, tablas y clips de cada tarea
```

## Tecnologías

- Python 3
- [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/) y [Pillow](https://python-pillow.org/)
- [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) y [seaborn](https://seaborn.pydata.org/) para métricas y matrices de confusión (Práctica 3)
- [Ultralytics YOLO](https://docs.ultralytics.com/) para detección, seguimiento y entrenamiento (Práctica 4)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) y [SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) (con [Transformers](https://huggingface.co/docs/transformers)) para OCR (Práctica 4)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) para los landmarks de la cara y las manos (Práctica 5)
- [TensorFlow / Keras](https://keras.io/) para el *transfer learning* de MobileNetV2, [scikit-image](https://scikit-image.org/) para los descriptores HOG y LBP, y [deepface](https://github.com/serengil/deepface) como referencia (Práctica 5)
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

Entorno `VC_P4` (el enunciado propone Python 3.9.5; los resultados se obtuvieron con Python 3.12):

```bash
conda create --name VC_P4 python=3.12
conda activate VC_P4
pip install ultralytics lap transformers scipy scikit-image python-bidi PyYAML Shapely pyclipper ninja
pip install --no-deps easyocr
```

EasyOCR se instala sin dependencias para que no sustituya `opencv-python` por `opencv-python-headless`, que eliminaría `cv2.imshow`.

Para entrenar con GPU hay que instalar PyTorch con CUDA siguiendo las [instrucciones oficiales](https://pytorch.org/get-started/locally/). Las imágenes del dataset de matrículas no se incluyen en el repositorio por su tamaño (~7 GB). Una vez descargadas en `PRACTICA_4/dataset/`, el script [`preparar_dataset.py`](PRACTICA_4/dataset/preparar_dataset.py) genera el reparto train/val/test, las etiquetas YOLO, el `data.yaml` y el ground truth.

### Práctica 5

Entorno `VC_P5` (Python 3.11.5, como indica el enunciado):

```bash
conda create --name VC_P5 python=3.11.5
conda activate VC_P5
pip install matplotlib mtcnn tensorflow deepface tf-keras mediapipe scikit-learn scikit-image pyarrow
```

MediaPipe y deepface instalan como dependencia **OpenCV 5**, que ya no puede leer los modelos Caffe del detector DNN. Para dejar una única versión 4.12:

```bash
pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless
pip install opencv-contrib-python-headless==4.12.0.88
pip install --force-reinstall --no-deps opencv-contrib-python==4.12.0.88
```

Los datasets no se incluyen en el repositorio: **DatabaseGender59x65** se descarga del campus virtual (para los ejemplos) y **FER-2013** de [HuggingFace](https://huggingface.co/datasets/AutumnQiu/fer2013/tree/main/data) (para la Tarea 1). Ambos van en `PRACTICA_5/data/`. TensorFlow no usa la GPU en Windows nativo, así que el entrenamiento se hace en CPU.

> Los demostradores en vivo y los prototipos de las prácticas 1, 2, 3 y 5 necesitan una webcam conectada.

## Autores

- **Pablo**, estudiante de Ingeniería Informática en la ULPGC.
- **Roberto**, estudiante de Ingeniería Informática en la ULPGC.
