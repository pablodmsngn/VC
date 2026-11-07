## Práctica 1. Primeros pasos con OpenCV

## Descripción

Primer contacto con OpenCV y NumPy: crear y modificar imágenes píxel a píxel, usar las funciones de dibujo, abrir vídeos y procesar en tiempo real la imagen de la webcam.

Todo el trabajo está en el notebook [`VC_P1.ipynb`](VC_P1.ipynb), que combina los ejemplos de la práctica con la resolución de las tareas.

---

## Tareas

### Tarea 1 — Tablero de ajedrez

Crear una imagen de 800×800 píxeles con la textura de un tablero de ajedrez, primero de forma manual y después con un asistente de IA, comparando ambas versiones.

### Tarea 2 — Imagen estilo Mondrian

Crear una composición al estilo de Piet Mondrian rellenando zonas rectangulares de una imagen en color con distintos colores.

### Tarea 3 — Píxel más claro y más oscuro en la webcam

En cada fotograma de la cámara se pasa a escala de grises y se localizan el píxel más oscuro y el más claro con `cv2.minMaxLoc`, dibujando un círculo sobre cada uno.

### Tarea 4 — Pop art propio

Efecto inspirado en el pop art de Andy Warhol sobre la imagen de la webcam:

* Se reduce la resolución del fotograma para que el procesado sea fluido.
* Se reduce el número de colores cuantizando cada canal (`(frame // 64) * 64`).
* Se compone un collage 2×2 con versiones del fotograma que resaltan cada canal de color.

---

## Técnicas utilizadas

* Creación y manipulación de imágenes como arrays de NumPy.
* Visualización con OpenCV (`imshow`) y Matplotlib.
* Funciones de dibujo de OpenCV.
* Lectura de vídeo desde archivo y desde la webcam (`cv2.VideoCapture`).
* Búsqueda de extremos de intensidad (`cv2.minMaxLoc`).
* Cuantización de color y composición de imágenes.

## Archivos

| Archivo | Contenido |
|---|---|
| [`VC_P1.ipynb`](VC_P1.ipynb) | Ejemplos y tareas de la práctica |
| `img/imagen.jpg` | Imagen generada y guardada desde el notebook |
| `videos/VermeerFun.mp4` | Vídeo de ejemplo para la lectura de vídeo desde archivo |

> Las tareas 3 y 4 necesitan una webcam conectada.
