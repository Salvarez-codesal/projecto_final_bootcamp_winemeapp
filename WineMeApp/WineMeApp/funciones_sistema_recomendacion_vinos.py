# Cargamos librerias
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
import re
import math
import matplotlib.pyplot as plt
from unidecode import unidecode
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import surprise
from surprise import BaselineOnly, Dataset, Reader
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import tensorflow_hub as hub

import numpy as np
from sklearn.manifold import TSNE

from scipy.stats import rankdata
data = pd.read_csv('WineMeApp/df_vinos_modelos.csv')


def obtener_embeddings(data):
    '''La función obtener_embeddings toma un DataFrame y una columna de texto como entrada, en nuestro caso, la columna con las notas de cata y de maridaje (`descripcion2`) calcula los embeddings de las descripciones de texto en esa columna utilizando el Universal Sentence Encoder (USE) de TensorFlow Hub y devuelve los embeddings resultantes como una matriz numpy.

    Input:
    - data: El DataFrame que contiene los datos de los vinos
    - columna_descripcion2: El nombre de la columna que contiene las descripciones de texto de los vinos.

    Output:
    - embeddings: Una matriz numpy que contiene los 512 embeddings/vectores de las descripciones de cata y maridaje. Cada fila de la matriz representa un vino y cada columna representa una dimensión en el espacio de embeddings.
    '''

    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    embeddings = []

    for descripcion in tqdm(data["descripcion2"].tolist()):
        embeddings.append(use([descripcion]))

    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape([embeddings.shape[0], embeddings.shape[2]])

    return embeddings
# _____
embeddings2 = obtener_embeddings(data)

# _____

def compute_scores(a, b):
    '''
    Esta función calcula la similitud de coseno entre dos conjuntos de vectores normalizados utilizando TensorFlow.

    Input: La función espera dos tensores a y b, donde cada uno representa un conjunto de vectores. Se espera que estos vectores ya estén normalizados en la dimensión 1.
    Output: puntuación de similitud para cada par de vectores de los conjuntos a y b. La puntuación se calcula como la similitud de coseno entre los vectores
    correspondientes en a y b. Se realiza un ajuste adicional para asegurarse de que las similitudes estén dentro del rango [-1, 1], y luego se convierten en puntajes de similitud en el rango [0, 1].
    '''
    a = tf.nn.l2_normalize(a, axis=1)
    b = tf.nn.l2_normalize(b, axis=1)
    cosine_similarities = tf.matmul(a, b, transpose_b=True)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    return scores

# _____

def get_recommendations(query, embeddings2, top_k=10):
    '''
    Esta función proporciona recomendaciones basadas en similitud de coseno entre un vector de consulta, vino que introducimos del dataset, y una matriz de vectores de incrustación (los embeddings)

    Input: La función espera tres argumentos:
        - query: Un vector de consulta, el vino que queramos buscar por posicion del index, que puede ser un único vector o una matriz de vectores
        - embeddings: Una matriz de vectores de incrustación donde se buscarán similitudes con el vector de consulta
        - top_k: El número de elementos más similares que se devolverán como recomendaciones (por defecto es 10), podemos modificarlo.
    '''
    if len(query.shape) == 1:
        query = tf.expand_dims(query, 0)
    sim_matrix = compute_scores(query, embeddings2)
    rank = tf.math.top_k(sim_matrix, top_k)
    return rank

# ____


# ---

def buscar_vino(data, texto_busqueda, embeddings):
    """
    Esta función busca un vino específico en el DataFrame de bodeboca.com de vinos de España y devuelve un DataFrame con los resultados de la búsqueda.

    Argumentos:
        - data (pandas.DataFrame): DataFrame que contiene información sobre los vinos, incluido el título del vino.
        - texto_busqueda (str): Nombre del vino que se desea buscar.

    Returns:
        - pandas.DataFrame: DataFrame que contiene información sobre los vinos encontrados, incluido el título. Si no se encuentra el vino buscado, se devuelve un DataFrame vacío.
    """

    # Normalizar el texto de búsqueda (eliminando acentos y convirtiendo a minúsculas)
    texto_busqueda = unidecode(texto_busqueda).lower()

    # Crear una expresión regular que coincida con cualquier texto que contenga las palabras en cualquier orden y con cualquier cantidad de palabras intermedias
    patron = '.*'.join(re.escape(word) for word in texto_busqueda.split())

    # Crear una máscara booleana que identifique las filas donde el texto está presente en la columna de interés
    mascara = data['titulo'].str.lower().apply(unidecode).str.contains(patron, flags=re.IGNORECASE)

    # Obtener los índices de las filas que cumplen con la condición
    indices = data[mascara].index

    # Calcular las recomendaciones utilizando los índices de los vinos encontrados
    scores, rank = get_recommendations(embeddings[indices, :], embeddings, top_k=15)
    recomendaciones = data.iloc[rank.numpy().reshape(-1).tolist(), :]

    # Usar la máscara para filtrar el DataFrame original y obtener las filas que cumplen con la condición
    resultados = data[mascara].reset_index()

    # Devolver el DataFrame de resultados
    return resultados

#____

def recomendar_vino(data, nombre_vino, embeddings):
    """
    Esta función calcula las recomendaciones de vinos similares para un vino específico en base a su nombre y devuelve un DataFrame con las recomendaciones.

    Argumentos:
        - data (pandas.DataFrame): DataFrame que contiene información sobre los vinos, incluido el título del vino.
        - embeddings (numpy.ndarray): Matriz de embeddings donde cada fila representa el embedding de un vino.
        - get_recommendations (callable): Función que recibe embeddings y devuelve recomendaciones basadas en similitud.
        - nombre_vino (str): Nombre del vino para el cual se desea obtener recomendaciones.

    Returns:
        - pandas.DataFrame: DataFrame que contiene información sobre los vinos recomendados, incluida su similitud. Si no se encuentra el vino buscado, se devuelve un DataFrame vacío.
    """
    # Buscar el vino y obtener el DataFrame de resultados
    resultados_busqueda = buscar_vino(data, nombre_vino, embeddings)

    # Obtener el índice del vino en el DataFrame de resultados
    indice_vino = resultados_busqueda.iloc[0][0]

    # Calcular las recomendaciones utilizando los índices de los vinos encontrados
    scores, rank = get_recommendations(embeddings[indice_vino,:], embeddings, top_k=11)
    recomendaciones = data.iloc[rank.numpy().reshape(-1).tolist(), :]

    # Crear una nueva columna en el DataFrame de recomendaciones para almacenar los scores de similitud de cada vino
    lista_de_listas = scores.numpy().tolist()
    lista_scores = [elemento for sublista in lista_de_listas for elemento in sublista]
    recomendaciones["Score_similitud"] = lista_scores
    recomendaciones = recomendaciones.loc[:, ["titulo", "vista", "nariz", "boca", "maridaje", "tipo", "tipo2", "origen", "variedad", "precio", "rating", "Score_similitud", "link"]]

    # Devolver el DataFrame de recomendaciones
    return recomendaciones

#######################################################################################################
#                                         -- TOP-SIS --
#######################################################################################################
precio_min = 5
precio_max = 150
# generar DataFrame al que aplicar topsis
def filtrar_por_precio(recomendaciones, precio_min, precio_max):
    """
  Función para filtrar un DataFrame por rango de precios.

  Argumentos:
    df: El DataFrame a filtrar.
    precio_min: Precio mínimo del rango.
    precio_max: Precio máximo del rango.

  Retorno:
    DataFrame filtrado por rango de precios.
  """

    # Filtrar por rango de precios
    df_filtrado = recomendaciones[(precio_min <= recomendaciones["precio"]) & (recomendaciones["precio"] <= precio_max)]

    vinos = df_filtrado[["titulo", "link", "tipo2", "precio", "rating"]]
    vinos = vinos.dropna()

    return vinos

def topsis(vinos):
    benefit_attributes = set([0, 1, 2, 3, 4])
    raw_data = vinos.values[:, 3:]
    candidates = vinos.values[:, 0]
    attributes = list(vinos.columns[3:])

    m = len(raw_data)  # calcula el numero de datos
    n = len(attributes)  # calcula el numero de atributos

    # Crear un array vacío para almacenar los divisores de normalización
    divisors = np.empty(n)
    for j in range(n):  # El bucle for itera a través de cada atributo (j):
        column = raw_data[:, j]  # Esto calcula la desviación estándar (raíz cuadrada de la varianza) de la column.
        divisors[j] = np.sqrt(column @ column)

    raw_data /= divisors  # Esto normaliza todo el array raw_data dividiendo cada elemento por el valor correspondiente en el array divisors (división elemento a elemento).

    column_names = ["X {}".format(i + 1) for i in range(n)]
    df_topsis = pd.DataFrame(data=raw_data, index=candidates, columns=column_names)
    #     print("Esta es la matriz normalizada:")
    df_topsis

    ######### 2.APLICAMOS LOS PESOS

    w_precios = 0.80  # peso de precios
    w_rating = 0.20  # peso de rating

    weights = np.array([w_precios, w_rating])

    raw_data *= weights
    df_topsis_pesos = pd.DataFrame(data=raw_data, index=candidates, columns=column_names)
    # Convierte el índice en una columna llamada "titulo"
    df_topsis_pesos.reset_index(inplace=True)
    df_topsis_pesos.rename(columns={"index": "titulo"}, inplace=True)
    df_topsis_pesos

    ######### 3.MATRIZ IDEAL /ANTIIDEAL

    a_pos = np.zeros(n)  # Este vector representará los valores positivos ideales
    a_neg = np.zeros(n)  # Este vector representará los valores negativos ideales

    for j in range(n):
        column = raw_data[:,
                 j]  # Extrae la columna j específica del array raw_data como un array separado llamado column.
        max_val = np.max(column)  # Calcula el valor máximo en la columna
        min_val = np.min(column)  # Calcula el valor mínimo en la columna

        # See if we want to maximize benefit or minimize cost (for PIS)
        if j in benefit_attributes:
            a_pos[j] = max_val
            a_neg[j] = min_val
        else:
            a_pos[j] = min_val
            a_neg[j] = max_val

    pd.DataFrame(data=[a_pos, a_neg], index=["$A^*$", "$A^-$"], columns=column_names)

    ######### 4.MATRIZ DISTANCIAS

    sp = np.zeros(m)
    sn = np.zeros(m)
    cs = np.zeros(m)

    for i in range(m):
        diff_pos = raw_data[i] - a_pos
        diff_neg = raw_data[i] - a_neg
        sp[i] = np.sqrt(diff_pos @ diff_pos)
        sn[i] = np.sqrt(diff_neg @ diff_neg)
        cs[i] = sn[i] / (sp[i] + sn[i])

    pd.DataFrame(data=zip(sp, sn, cs), index=candidates, columns=["$S^*$", "$S^-$", "$C^*$"])

    ######### 5.MATRIZ DISTANCIAS ORDENADA POR RANKING

    # Define una funcion para asignar un orden de ranking
    def rank_according_to(data):
        ranks = rankdata(data).astype(int)
        ranks -= 1
        return candidates[ranks][::-1]

    cs_order = rank_according_to(cs)
    sp_order = rank_according_to(sp)
    sn_order = rank_according_to(sn)

    df_ranking = pd.DataFrame(data=zip(cs_order, sp_order, sn_order), index=range(1, m + 1),
                              columns=["$C^*$", "$S^*$", "$S^-$"])
    # Convierte el índice en una columna llamada "ranking_topsis"
    df_ranking.reset_index(inplace=True)
    df_ranking.rename(columns={"index": "ranking_topsis"}, inplace=True)

    return df_topsis_pesos, df_ranking

def visualizacion_interactiva(df):
    """
    Crea un diagrama de dispersión para visualizar datos de vinos con diferentes opciones de personalización.

    Parámetros:
        df (pandas.DataFrame): El DataFrame que contiene los datos de los vinos.

    Devoluciones:
        plotly.Figure: El objeto de figura de Plotly que representa el diagrama de dispersión.
    """

    # Obtiene las columnas para el eje X y el eje Y
    x = df["X 1"]
    y = df["X 2"]

    # Definimos un mapeo de colores para los tipos de vinos
    color_map = {
        'tinto': 'crimson',
        'blanco': 'sandybrown',
        'espumoso': 'royalblue',
        'generoso': 'purple',
        'rosado': 'violet',
        'vermouth': 'limegreen'}

    # Define el tamaño de los puntos en función del ranking TOPSIS
    sizes = (df["ranking_topsis"].max() - df["ranking_topsis"]) / df["ranking_topsis"].max() * 20 + 5

    # Define el gráfico
    fig = go.Figure()

    # Agrega los puntos al gráfico
    for tipo in color_map.keys():
        datos_tipo = df[df["tipo2"] == tipo]
        fig.add_scatter(
            x=datos_tipo["X 1"],  # Corregido aquí
            y=datos_tipo["X 2"],  # Corregido aquí
            name=tipo,
            mode="markers",
            marker=dict(color=color_map[tipo], size=sizes, sizemode='diameter'),
            hoverinfo="text",
            text="**Tipo:** " + tipo + "<br>" + datos_tipo["titulo"] + "<br>Ranking TOPSIS: " + datos_tipo["ranking_topsis"].astype(str) + "<br>Precio: "+ datos_tipo["precio"].astype(str),  # Corregido aquí
        )

    # Personaliza el título y las etiquetas de los ejes
    fig.update_layout(
        xaxis_title="Precio_ponderado",
        yaxis_title="Ranking_ponderado",
        legend_title="Tipos de Vinos",
    )

    # Muestra el gráfico y devuelve la figura
    return fig
# ------------------------------------------------------------

def visualizacion_interactiva2(df):
  # Obtiene las columnas para el eje X y el eje Y
    x = df["X 1"]
    y = df["X 2"]

    # Definimos un mapeo de colores para los tipos de vinos
    color_map = {
    'tinto': 'crimson',
    'blanco': 'sandybrown',
    'espumoso': 'royalblue',
    'generoso': 'purple',
    'rosado': 'violet',
    'vermouth': 'limegreen'}

    # Crea un diccionario con los datos
    datos = {"x": x, "y": y, "tipo2": df["tipo2"]}

    # Define el gráfico
    fig = go.Figure()

    # Agrega los puntos al gráfico
    fig.add_scatter(
    x=datos["x"],
    y=datos["y"],
    name="Vinos",
    mode="markers",
    marker=dict(color=df["tipo2"].map({'tinto': 'crimson','blanco': 'sandybrown',
                            'espumoso': 'royalblue','generoso': 'purple',
                            'rosado': 'violet', 'vermouth': 'limegreen'}),size=15),
    hoverinfo="text",
    text=df["titulo"],
    )

    # Personaliza el título y las etiquetas de los ejes
    fig.update_layout(
    title="Diagrama de dispersión de vinos",
    xaxis_title="X1",
    yaxis_title="Y1",
    )


    # Muestra el gráfico
    return fig.show()

# ------------------------------------------------------------
def topsis_general(recomendaciones, nombre_vino):
    recomendaciones = recomendar_vino(data, nombre_vino, embeddings2)[1:]
    vinos = filtrar_por_precio(recomendaciones, precio_min, precio_max)
    df_topsis_pesos, df_ranking = topsis(vinos)
    df_ranking.rename(columns={"$C^*$": "titulo"}, inplace=True)
    df_combinado = vinos.merge(df_ranking[["ranking_topsis","titulo"]], how='left', on='titulo')
    df_combinado = df_combinado.sort_values(by='ranking_topsis', ascending=True)  # Ordenar por 'ranking_topsis' de forma descendente
    df_combinado2 = df_combinado.merge(df_topsis_pesos[["titulo", "X 1", "X 2"]], how='left', on='titulo')
    return  visualizacion_interactiva(df_combinado2)

