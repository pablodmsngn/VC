# Prácticas de Visión por Computador

Repositorio con las prácticas realizadas en la asignatura **Visión por Computador** del Grado en **Ingeniería Informática** de la **Universidad de Las Palmas de Gran Canaria (ULPGC)**.

Cada práctica aborda un bloque distinto de la asignatura y se entrega como un notebook de Jupyter con su código y explicaciones.

## Contenido

| Práctica | Tema | Archivo |
|----------|------|---------|
| Práctica 1 | Ajedrez | 'VC_P1.ipynb' |
| Práctica 2 | Detección de bordes (Canny y Sobel) y demostrador con webcam | `VC_P2_TAREAS.ipynb` |
| Práctica 3 | Detección y reconocimiento de formas: conteo de monedas y clasificación de microplásticos | `TAREAS/VC_P3_TAREA_1.ipynb`, `TAREAS/VC_P3_TAREA_2.ipynb` ([README](PRACTICA_3/README-P3.md)) |


## Tecnologías

- Python 3
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow (PIL)](https://python-pillow.org/)
- [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/) y [seaborn](https://seaborn.pydata.org/) (métricas y matriz de confusión de la Práctica 3)
- Jupyter Notebook

## Instalación y uso

```bash
# Clonar el repositorio
git clone <URL-DEL-REPOSITORIO>
cd <NOMBRE-DEL-REPOSITORIO>

# (Opcional) crear un entorno virtual
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install opencv-python numpy matplotlib pillow jupyter pandas scikit-learn seaborn

# Abrir los notebooks
jupyter notebook
```

> El demostrador de las Practicas se necesita una webcam conectada.


## Autor

**Pablo**, estudiante de Ingeniería Informática en la ULPGC.
**Roberto**, estudiante de Ingeniería Informática en la ULPGC.
