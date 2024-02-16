import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Task_1.unsupervised.src import svd  # Importamos el módulo svd desde nuestro paquete

# Paso 1: Leer la imagen
image_path = "C:/Users/DO IT WISER/Documents/Personal/machine_learning2/Task_Photos/Melissa_Arevalo_1.jpeg"
image = Image.open(image_path)

# Paso 2: Convertir la imagen a escala de grises
image_gray = image.convert('L')
image_array = np.array(image_gray)

# Paso 3: Aplicar el SVD a la imagen
U, s, V = svd(image_array)

# Paso 4 y 5: Reconstruir la imagen con diferentes números de valores singulares y evaluar la calidad
num_singular_values = 100  # Número de valores singulares a utilizar inicialmente
plt.figure(figsize=(12, 6))
for i in range(1, 10):
    reconstructed_image = np.dot(U[:, :num_singular_values] * s[:num_singular_values], V[:num_singular_values, :])
    plt.subplot(3, 3, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title("Número de valores singulares: {}".format(num_singular_values))
    plt.axis('off')
    num_singular_values += 100

plt.tight_layout()
plt.show()