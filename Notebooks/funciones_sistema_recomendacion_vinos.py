# Funciones para el modelo de recomendacion de vinos
# Autores: Ivan Pinto Grilo, Maria Perez Sebastian, Soraya Alvarez Codesal
# Fecha modificacion: 18/04/24

# Cargamos librerias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from unidecode import unidecode  
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import surprise
from surprise import BaselineOnly, Dataset, Reader
import warnings
warnings.filterwarnings('ignore')
import cloudpickle
from tqdm import tqdm
import tensorflow_hub as hub
import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf 
import math 


def obtener_embeddings(data, columna_descripcion2):
    '''La función obtener_embeddings toma un DataFrame y una columna de texto como entrada, en nuestro caso, la columna con las notas de cata y de maridaje (`descripcion2`) calcula los embeddings de las descripciones de texto en esa columna utilizando el Universal Sentence Encoder (USE) de TensorFlow Hub y devuelve los embeddings resultantes como una matriz numpy.
    
    Input:
    - data: El DataFrame que contiene los datos de los vinos
    - columna_descripcion2: El nombre de la columna que contiene las descripciones de texto de los vinos.
    
    Output:
    - embeddings: Una matriz numpy que contiene los 512 embeddings/vectores de las descripciones de cata y maridaje. Cada fila de la matriz representa un vino y cada columna representa una dimensión en el espacio de embeddings.
    '''
    
    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = []

    for descripcion in tqdm(data[columna_descripcion2].tolist()):
        embeddings.append(use([descripcion]))

    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape([embeddings.shape[0], embeddings.shape[2]])
    
    return embeddings



# creamos las funciones para el motor de recomendacion
def compute_scores(a, b):
    '''
    Esta función calcula la similitud de coseno entre dos conjuntos de vectores normalizados utilizando TensorFlow. 
    
    Input: La función espera dos tensores a y b, donde cada uno representa un conjunto de vectores. Se espera que estos vectores ya estén normalizados en la dimensión 1.
    Output: puntuación de similitud para cada par de vectores de los conjuntos a y b. La puntuación se calcula como la similitud de coseno entre los vectores
    correspondientes en a y b. Se realiza un ajuste adicional para asegurarse de que las similitudes estén dentro del rango [-1, 1], y luego se convierten en puntajes de similitud en el rango [0, 1].
    '''
    a = tf.nn.l2_normalize(a, axis=1)
    b = tf.nn.l2_normalize(b, axis=1)
    cosine_similarities = tf.matmul(a, b, transpose_b = True)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    return scores


def get_recommendations(query, embeddings, top_k = 10):
    '''
    Esta función proporciona recomendaciones basadas en similitud de coseno entre un vector de consulta, vino que introducimos del dataset, y una matriz de vectores de incrustación (los embeddings)
    
    Input: La función espera tres argumentos:
        - query: Un vector de consulta, el vino que queramos buscar por posicion del index, que puede ser un único vector o una matriz de vectores
        - embeddings: Una matriz de vectores de incrustación donde se buscarán similitudes con el vector de consulta
        - top_k: El número de elementos más similares que se devolverán como recomendaciones (por defecto es 10), podemos modificarlo.
    '''
    if len(query.shape)==1:
        query = tf.expand_dims(query, 0)
    sim_matrix = compute_scores(query, embeddings)
    rank = tf.math.top_k(sim_matrix, top_k)
    return rank



# FUNCION DE RECOMENDACION DE VINOS
def buscar_vino_y_recomendar(data, EMBEDDINGS, get_recommendations):
    """
    Esta función busca un vino específico en el DataFrame de bodeboca.com de vinos de España y luego recomienda vinos similares
    basados en los embeddings de los vinos encontrados. top_k=n determina los numero de vinos (n) que queremos mostrar como resultado del recomendador.
    Por defecto, si no se pone nada seria top_k = 10

    Argumentos:
        - data (pandas.DataFrame): DataFrame que contiene información sobre los vinos, incluido el título del vino.
        - EMBEDDINGS (numpy.ndarray): Matriz de embeddings donde cada fila representa el embedding de un vino.
        - get_recommendations (callable): Función que recibe embeddings y devuelve recomendaciones basadas en similitud.

    Returns:
        - pandas.DataFrame: DataFrame que contiene información sobre los vinos recomendados, incluida su similitud. El primer vino siempre es el que
        introduce el usuario. Si no se encuentra el vino buscado, se devuelve un DataFrame vacío.
    """
    # Solicitar al usuario que ingrese el vino que desea verificar
    texto_busqueda = input("Ingrese el nombre del vino que desea buscar: ")

    # Normalizar el texto de búsqueda (eliminando acentos y convirtiendo a minúsculas)
    texto_busqueda = unidecode(texto_busqueda).lower()

    # Crear una expresión regular que coincida con cualquier texto que contenga las palabras en cualquier orden y con cualquier cantidad de palabras intermedias
    patron = '.*'.join(re.escape(word) for word in texto_busqueda.split())

    # Crear una máscara booleana que identifique las filas donde el texto está presente en la columna de interés
    mascara = data['titulo'].str.lower().apply(unidecode).str.contains(patron, flags=re.IGNORECASE)

    # Usar la máscara para filtrar el DataFrame original y obtener las filas que cumplen con la condición
    resultados = data[mascara].reset_index()

    # Imprimir los resultados
    if len(resultados) > 0:
        print("Hemos encontrado", len(resultados), "vino/s en nuestro sistema")
        print("Resultados de la búsqueda:")
        print(resultados.titulo)
        
        # Solicitar al usuario que ingrese el vino para el cual desea buscar una recomendación
        vino = input("Por favor, ingrese el número del vino para el cual desea buscar una recomendación: ")

        # Normalizar el texto de búsqueda (eliminando acentos y convirtiendo a minúsculas)
        vino = unidecode(vino).lower()

        # Crear una expresión regular que coincida con cualquier texto que contenga las palabras en cualquier orden y con cualquier cantidad de palabras intermedias
        patron2 = '.*'.join(re.escape(word) for word in vino.split())

        # Crear una máscara booleana que identifique las filas donde el texto está presente en la columna de interés
        mascara = data['titulo'].str.lower().apply(unidecode).str.contains(patron2, flags=re.IGNORECASE)

        # Obtener los índices de las filas que cumplen con la condición
        indices = data[mascara].index

        # Calcular las recomendaciones utilizando los índices de los vinos encontrados
        scores, rank = get_recommendations(EMBEDDINGS[indices,:], EMBEDDINGS, top_k=8)
        recomendaciones = data.iloc[rank.numpy().reshape(-1).tolist(), :]

        # Crear una nueva columna en el DataFrame de recomendaciones para almacenar los scores de similitud de cada vino
        lista_de_listas = scores.numpy().tolist()
        lista_scores = [elemento for sublista in lista_de_listas for elemento in sublista]
        recomendaciones["Score_similitud"] = lista_scores

        # Imprimir la selección de vinos con información adicional
        print("Los vinos más afines a su vino son los siguientes:")
        # print(recomendaciones)
        
        # Devolver el DataFrame de recomendaciones como resultado de la función
        return recomendaciones

    else:
        print("Su vino no se encuentra en nuestra base de datos")



        
## Si queremos utilizar las funciones del recomendador de vinos se haria como acontinuacion
# import pandas as pd
# from funciones_sistema_recomendacion_vinos import *

# # Cargar el DataFrame de vinos desde un archivo CSV o desde donde lo tengas
# data = pd.read_csv('df_vinos_modelos.csv')

# # Obtener los embeddings de las descripciones de cata y maridaje
# EMBEDDINGS = obtener_embeddings(data, "descripcion2")

# # Llamar a la función para buscar y recomendar vino
# resultado_recomendaciones = buscar_vino_y_recomendar(data, EMBEDDINGS, get_recommendations)

# Puedes hacer lo que quieras con el resultado, como guardarlo en un archivo CSV
#resultado_recomendaciones.to_csv('recomendaciones_vinos.csv', index=False)

        
