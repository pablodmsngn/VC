# 

# Guías para anotar las imágenes

## Estructura de carpetas

Existen 2 carpetas en el dataset:

- `images`: imágenes de coches  
- `annotations`: anotaciones de las matrículas para cada imagen

## Guía de uso

Estos son los siguientes pasos necesarios para anotar una imagen:

1. Poner un nombre a la imagen. **Todas las imágenes deberán subirse en formato `jpg`**. La imagen se deberá almacenar en la carpeta `images`.  
2. Anotar la imagen, con el formato YOLO (**COORDENADAS NORMALIZADAS**). Este es el formato a seguir (suponer que la imagen tiene 2 anotaciones, una de ellas con matrícula legible, y otra ilegible):

```json
[
    {
        rectangle: [x_center, y_center, width, height],
        plate: "0000AAA"
    },
    {
        rectangle: [x_center, y_center, width, height],
        plate: "None"
    }
]
```

*Para anotar la imagen, pueden usar la herramienta que quieran, siempre y cuando cumplan el formato especificado. El script de Python que se encuentra en esta carpeta también les puede ayudar a anotar si lo necesitan. Probablemente tengan que instalar algún paquete si quieren usarlo. Tienen las instrucciones de uso como comentario sobre el propio script.*

3. Guardar la anotación en formato json y **con el mismo nombre que la imagen correspondiente**. Por ejemplo, si la imagen se llama `abc.jpg`, las anotaciones de la imagen estarán en `abc.json`. Almacenar la anotación en la carpeta `annotations`.

## Advertencias

- No incluir *data augmentation* como parte de las imágenes del dataset. Solo se guarda en `images` 1 vez cada imagen, realizando las modificaciones/transformaciones a posteriori en el código.  
- No reemplazar las imágenes (tengan cuidado de poner nombres únicos, por favor).  
- Por favor (y aunque sea obvio), anoten con conciencia ;)