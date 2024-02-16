import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

path = r"C:\Users\DO IT WISER\Documents\Personal\machine_learning2\Fotos ML2\Melissa_Arevalo_.jpg"

# Black and white image
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# Image dimensions
height, wide = image.shape

# Inicializa la variable para almacenar la traza
trace = 0

# Calcula la traza sumando los valores de los píxeles en la diagonal principal
for i in range(min(height, wide)):
    trace += image[i, i]

print("Image trace:", trace)


def cargar_imagenes(directorio):
    imagenes = []
    for filename in os.listdir(directorio):
        path = os.path.join(directorio, filename)
        if os.path.isfile(path):
            imagen = cv2.imread(path)
            if imagen is not None:  # Comprobar si la imagen se cargó correctamente
                if imagen.shape[:2] != (256, 256):  # Comprobar el tamaño de la imagen
                    imagen = cv2.resize(imagen, (256, 256))  # Redimensionar la imagen si es necesario
                imagenes.append(imagen)
            else:
                print(f"No se pudo cargar la imagen: {path}")
    return imagenes

def cara_promedio(imagenes):
    suma_caras = np.zeros_like(imagenes[0], dtype=np.float64)
    for imagen in imagenes:
        suma_caras += imagen.astype(np.float64)
    return (suma_caras / len(imagenes)).astype(np.uint8)

def mostrar_imagen(imagen):
    plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Directorio que contiene las imágenes de los rostros
directorio = r'C:\Users\DO IT WISER\Documents\Personal\machine_learning2\Fotos ML2'

# Cargar las imágenes
imagenes = cargar_imagenes(directorio)

# Calcular la cara promedio
cara_promedio_img = cara_promedio(imagenes)

ruta_guardado_promedio = r'C:\Users\DO IT WISER\Documents\Personal\machine_learning2\cara_promedio.png'
cv2.imwrite(ruta_guardado_promedio, cara_promedio_img)

# Mostrar la cara promedio
mostrar_imagen(cara_promedio_img)

def distancia_entre_imagenes(img1, img2):
    # Convertir las imágenes a vectores de características
    vec1 = img1.flatten()
    vec2 = img2.flatten()

    # Calcular la distancia euclidiana entre los vectores de características
    distancia = np.linalg.norm(vec1 - vec2)
    return distancia

def cargar_imagen(ruta):
    imagen = cv2.imread(ruta)
    if imagen is None:
        print(f"No se pudo cargar la imagen en {ruta}")
    return imagen

# Cargar la imagen específica y la cara promedio
ruta_imagen_especifica = r'C:\Users\DO IT WISER\Documents\Personal\machine_learning2\Fotos ML2\Melissa_Arevalo_.jpg'
ruta_cara_promedio = r'C:\Users\DO IT WISER\Documents\Personal\machine_learning2\cara_promedio.png'

imagen_especifica = cargar_imagen(ruta_imagen_especifica)
cara_promedio = cargar_imagen(ruta_cara_promedio)

# Calcular la distancia entre la imagen específica y la cara promedio
distancia = distancia_entre_imagenes(imagen_especifica, cara_promedio)
print("La distancia entre la imagen específica y la cara promedio es:", distancia)