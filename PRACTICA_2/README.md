## Práctica 2. Funciones básicas de OpenCV

## Descripción

El objetivo es analizar imágenes mediante detección de bordes, umbralización y conteo de píxeles, para posteriormente aplicar estos conceptos en un demostrador interactivo basado en vídeo.

La práctica se divide en tres tareas:

* **Tarea 1:** análisis de filas mediante Canny.
* **Tarea 2:** análisis de filas y columnas mediante Sobel y Canny.
* **Tarea 3:** creación de un demostrador de privacidad digital basado en sustracción de fondo.

---

## Tarea 1 — Análisis de filas mediante Canny

En esta tarea se aplica el operador **Canny** sobre la imagen del mandril para obtener una imagen binaria de los bordes.

Posteriormente se realiza un conteo de los píxeles blancos de cada fila mediante `cv2.reduce()`. A partir de este conteo se obtiene el número máximo de píxeles blancos presentes en una fila (`maxfil`).

Finalmente, se seleccionan las filas cuyo número de píxeles blancos sea igual o superior al **90 % del máximo** y se resaltan mediante líneas rojas sobre la imagen de Canny.

### Técnicas utilizadas

* Conversión de BGR a escala de grises.
* Detección de bordes mediante Canny.
* Conteo de píxeles por filas.
* Obtención del máximo de píxeles blancos.
* Selección de filas mediante un umbral del 90 %.
* Representación gráfica de las filas seleccionadas.

---

## Tarea 2 — Análisis de Sobel y comparación con Canny

En esta tarea se utiliza el operador **Sobel** para detectar cambios de intensidad en la imagen tanto en dirección horizontal como vertical.

Los resultados obtenidos mediante Sobel se combinan y posteriormente se convierten a una imagen de 8 bits. Debido a que Sobel produce diferentes niveles de intensidad, se aplica un umbral para obtener una imagen binaria de los bordes.

A continuación, se realiza un análisis por filas y columnas similar al realizado con Canny. Se obtiene el máximo de píxeles por fila y columna y se seleccionan aquellas que superan el **90 % del máximo**. Las filas se representan mediante líneas rojas y las columnas mediante líneas verdes.

También se realiza el mismo análisis sobre la imagen obtenida mediante Canny para poder comparar ambos métodos.

### Comparación entre Sobel y Canny

Sobel detecta los bordes a partir de los cambios de intensidad entre píxeles, calculando dichos cambios en las direcciones horizontal y vertical. El resultado contiene diferentes niveles de intensidad, por lo que es necesario convertirlo a 8 bits y posteriormente aplicar un umbral para obtener una imagen binaria de los bordes.

Canny también utiliza información relacionada con el gradiente de la imagen, pero realiza un procesamiento adicional para obtener bordes más definidos, finos y conectados. Además, utiliza dos umbrales para diferenciar entre bordes fuertes y débiles. Cabe destacar que ese proceso lo hace de manera interna, mientras que con sobel tenemos que hacerlo manualmente.

Al comparar ambos resultados, se observa que Sobel umbralizado y Canny detectan los bordes de forma diferente. Canny suele producir bordes más definidos y precisos, mientras que el resultado de Sobel depende en gran medida del valor de umbral seleccionado. Un umbral más alto puede hacer que Sobel detecte menos bordes, mientras que uno más bajo puede hacer que detecte más.

En ambos casos, el análisis de las filas y columnas permite localizar las zonas de la imagen donde se concentra una mayor cantidad de píxeles correspondientes a bordes. En este caso concreto, se observa una mayor cantidad de píxeles de borde en el resultado obtenido mediante Canny.

### Técnicas utilizadas

* Filtro Gaussiano.
* Detección de bordes con Sobel.
* Conversión a 8 bits y umbralización.
* Análisis por filas y columnas.
* Comparación con Canny.

---

## Tarea 3 — Demostrador: Cortina de privacidad digital

Como reinterpretación de la instalación **My Little Piece of Privacy**, se desarrolla un demostrador interactivo que utiliza la cámara del ordenador para crear una cortina digital de privacidad.

Para ello se utiliza `BackgroundSubtractorMOG2`, que realiza una **sustracción de fondo** para detectar las regiones que presentan cambios respecto al fondo aprendido.

El resultado de MOG2 puede distinguir entre fondo, sombras y objetos detectados. Para eliminar las sombras se aplica posteriormente una **umbralización**, conservando únicamente las regiones correspondientes a los objetos detectados.

A partir de la máscara obtenida se cuenta el número de píxeles blancos de cada columna. Se obtiene la columna con mayor número de píxeles detectados y se utilizan las columnas que superan un porcentaje determinado de dicho máximo para establecer los límites de la persona detectada.

Finalmente, se dibuja un rectángulo vertical que cubre desde la primera hasta la última columna seleccionada, creando una **cortina digital** que oculta la zona donde se encuentra la persona.


### Técnicas utilizadas

* Captura de vídeo mediante OpenCV.
* Sustracción de fondo mediante MOG2.
* Detección y eliminación de sombras.
* Umbralización binaria.
* Conteo de píxeles por columnas.
* Detección de los límites de la región ocupada.
* Representación gráfica mediante un rectángulo.

---

## Conclusión

Las tareas muestran distintas formas de procesar esta información: mediante la detección de bordes con Canny y Sobel, la umbralización y el análisis de píxeles por filas y columnas, así como mediante la sustracción de fondo aplicada al vídeo.

Finalmente, estos conceptos se aplican de forma conjunta en el demostrador de la cortina de privacidad digital, donde la información obtenida de la cámara se procesa en tiempo real para identificar la zona ocupada por una persona y generar una respuesta visual sobre la imagen.