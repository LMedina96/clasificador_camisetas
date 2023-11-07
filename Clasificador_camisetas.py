import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model, load_model
import tensorflow_hub as hub
import numpy as np
import scipy
from PIL import Image
import requests
from io import BytesIO
import cv2
import json

modelo_entrenado = any

#Tomamos el path de las carpetas de entrenamiento y hacer conteo
def conteoArchivos():
    boca_juniors_dir = "train_images\BocaJuniors"
    river_plate_dir = "train_images\RiverPlate"
    racing_dir = "train_images\Racing"
    
    #Definimos la opción para el conteo de camisetas
    def count_files_in_directory(directory):
        if os.path.exists(directory):
            files = os.listdir(directory)
            count = len(files)
            return count
        else:
            return 0

    boca_juniors_count = count_files_in_directory(boca_juniors_dir)
    river_plate_count = count_files_in_directory(river_plate_dir)
    racing_count = count_files_in_directory(racing_dir)

    #Imprimimos el conteo
    print(f"Número de archivos en BocaJuniors: {boca_juniors_count}")
    print(f"Número de archivos en RiverPlate: {river_plate_count}")
    print(f"Número de archivos en Racing: {racing_count}")

################################## COPIAR A DATASET ###########################

def copiarADataset():
    # Definir las carpetas de origen y destino base
    base_fuente = 'train_images'
    carpeta_destino = 'dataset'

    # Lista de carpetas de equipos
    equipos = ['BocaJuniors', 'RiverPlate', 'Racing']

    # Iterar a través de las carpetas de equipos
    for equipo in equipos:
        carpeta_fuente = os.path.join(base_fuente, equipo)

        # Obtener la lista de imágenes en la carpeta de origen
        imagenes = os.listdir(carpeta_fuente)

        # Iterar a través de las imágenes y copiarlas a la carpeta de destino
        for nombreimg in imagenes:
            # Combinar las rutas de origen y destino
            origen_img = os.path.join(carpeta_fuente, nombreimg)
            destino_img = os.path.join(carpeta_destino, equipo, nombreimg)

            # Asegurarse de que la carpeta de destino exista
            os.makedirs(os.path.dirname(destino_img), exist_ok=True)

            # Copiar la imagen
            shutil.copy(origen_img, destino_img)
        
################################## AUMENTO DE DATOS ###########################

def aumentarDatos():
    #Crear el dataset generador
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range = 30,
        width_shift_range = 0.25,
        height_shift_range = 0.25,
        shear_range = 15,
        zoom_range = [0.5, 1.5],
        validation_split=0.2  # 20% para pruebas
    )

    # Generadores para sets de entrenamiento y pruebas
    data_gen_entrenamiento = datagen.flow_from_directory('dataset', target_size=(224,224), batch_size=32, shuffle=True, subset='training')
    data_gen_pruebas = datagen.flow_from_directory('dataset', target_size=(224,224), batch_size=32, shuffle=True, subset='validation')

    # Imprimir 10 imágenes del generador de entrenamiento
    for imagen, etiqueta in data_gen_entrenamiento:
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(imagen[i])
        break
    plt.show()

    return data_gen_entrenamiento, data_gen_pruebas

def entrenarModelo():
    # Llamar a la función aumentarDatos para obtener los generadores de datos
    data_entrenamiento, data_pruebas = aumentarDatos()

    url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3))

    # Congelar el modelo descargado
    mobilenetv2.trainable = False

    modelo_entrenado = tf.keras.Sequential([
        mobilenetv2,
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    modelo_entrenado.summary()

    # Compilamos
    modelo_entrenado.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    EPOCAS = 1

    historial_entrenamiento = modelo_entrenado.fit(
        data_entrenamiento, epochs=EPOCAS, batch_size=32,
        validation_data=data_pruebas
    )
    
    save_model(modelo_entrenado, "modelo_entrenado.keras")
    
    with open('historial_entrenamiento.json', 'w') as file:
        json.dump(historial_entrenamiento.history, file)
    
#Graficas de precisión
def graficarPresicion(historial):
    acc = historial['accuracy']
    val_acc = historial['val_accuracy']

    loss = historial['loss']
    val_loss = historial['val_loss']

    rango_epocas = range(len(acc))

    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(rango_epocas, acc, label='Precisión Entrenamiento')
    plt.plot(rango_epocas, val_acc, label='Precisión Pruebas')
    plt.legend(loc='lower right')
    plt.title('Precisión de entrenamiento y pruebas')

    plt.subplot(1,2,2)
    plt.plot(rango_epocas, loss, label='Pérdida de entrenamiento')
    plt.plot(rango_epocas, val_loss, label='Pérdida de pruebas')
    plt.legend(loc='upper right')
    plt.title('Pérdida de entrenamiento y pruebas')
    plt.show()

#Categorizar una imagen de internet
def categorizarImagen(modelo):
    def categorizar(url):
        respuesta = requests.get(url)
        img = Image.open(BytesIO(respuesta.content))
        img = np.array(img).astype(float)/255

        img = cv2.resize(img, (224,224))
        prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
        print(prediccion)
        return np.argmax(prediccion[0], axis=-1)

    url = r"https://www.opensports.com.ar/media/catalog/product/cache/4769e4d9f3516e60f2b4303f8e5014a8/G/I/GI4683_0.jpg"
    prediccion = categorizar(url)
    print(prediccion)


#Hacemos un conteo de los archivos de cada carpeta
#conteoArchivos()

#Creamos el dataset tomando las carpetas de entrenamiento
#copiarADataset()

#Mostramos algunas imagenes y entrenamos el modelo ¡Tarda unos minutos!
entrenarModelo()

#Graficamos la presición del entrenamiento
#with open('historial_entrenamiento.json', 'r') as file:
#    historial_cargado = json.load(file)
#graficarPresicion(historial_cargado)

#Categorizamos la imagen según el modelo que usamos de entrenamiento
modelo_entrenado = load_model("modelo_entrenado.keras")
categorizarImagen(modelo_entrenado)