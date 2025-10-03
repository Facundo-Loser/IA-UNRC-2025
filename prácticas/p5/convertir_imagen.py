from PIL import Image, ImageOps
import numpy as np
import network  # tu red neuronal
import matplotlib.pyplot as plt

# ---------- CARGAR Y PROCESAR IMAGEN NUEVA ---------- #
def procesar_imagen(ruta_imagen):
    # Cargar imagen en escala de grises
    img = Image.open(ruta_imagen).convert('L')

    # Invertir colores: MNIST tiene fondo negro y dígito blanco
    img = ImageOps.invert(img)

    # Redimensionar a 28x28 píxeles
    img = img.resize((28, 28))

    # Mostrar imagen para verificar
    plt.imshow(img, cmap='gray')
    plt.title("Imagen procesada")
    plt.show()

    # Convertir a arreglo numpy y normalizar
    img_array = np.array(img).astype(np.float32) / 255.0

    # Aplanar la imagen a vector de 784x1
    input_vector = img_array.reshape(784, 1)

    return input_vector

# ---------- USAR RED ENTRENADA PARA PREDECIR ---------- #
def predecir_imagen(net, imagen_path):
    input_vector = procesar_imagen(imagen_path)
    output = net.feedforward(input_vector)
    prediccion = np.argmax(output)
    print(f"Predicción de la red: {prediccion}")
    return prediccion

# ---------- EJECUCIÓN PRINCIPAL ---------- #
if __name__ == '__main__':
    # Cargar la red entrenada (o volver a entrenarla si no la guardaste)
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # Cargar red con pesos ya entrenados (acá se entrena de nuevo, pero podrías cargarla si la guardaste)
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 10, 10, 3.0)  # Entrena la red - podés comentar esto si ya la tenés entrenada

    # Probar con una imagen externa
    predecir_imagen(net, 'mi_numero.png')  # Ruta a tu imagen
