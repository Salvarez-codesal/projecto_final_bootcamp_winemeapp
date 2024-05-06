# FUNCIONES DE PREPROCESAMIENTO DATOS EXTRAIDOS DE BODEBOCA
# Autores: Ivan Pinto Grilo, Maria Perez Sebastian, Soraya Alvarez Codesal
# Fecha modificacion: 15/04/24


# Cargamos librerias
import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import warnings
warnings.filterwarnings('ignore')




# Función limpieza de texto
def limpiar_texto(texto):
    '''
    Esta funcion recibe un texto o un dataset y limpia los simbolos que representan letras con acentos o simbolos especiales por los correctos.
    
    Input: texto o dataset
    Output: texto o dataset limpio de simbolos raros por letras apropiadas
    '''
    if isinstance(texto, str):  # Verifica si el valor es una cadena de texto
        texto = texto.replace('Ã¡', 'a').replace('Ã©', 'e').replace('Ã-', 'i').replace('Ã³', 'o').replace('Ãº', 'u').replace('Ã±', 'ñ').replace('Ã¨', 'e').replace('Âº', 'º').replace('Â', 'i').replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'e').replace('ú', 'u').replace('ã', 'i')
        texto = texto.lower().strip()
    return texto


# funcion que extrae las palabras de un texto o columna - especialmente para variedad
def extraer_palabras(texto):
    '''
    Esta funcion recibe un texto y extrae solamente las palabras, usa una expresión regular para encontrar palabras en el texto y elimina los porcentajes
    
    Input: texto con simbolos, numeros ... etc
    Output: texto exclusivamente con letras
    '''
    texto_limpio = re.sub(r'[^a-zA-ZñÑ,\s]', '', texto)
    return texto_limpio


# funcion para Llenar los valores nulos en una columna con la moda de ese grupo
def fillna_mode(x):
    '''
    Esta función, se utiliza para rellenar los valores faltantes en una serie de datos con la moda de esa serie. 
    Si hay más de un valor no nulo en el grupo, se calcula la moda y se rellenan los valores faltantes con él. 
    Si solo hay un valor no nulo o ningún valor no nulo en el grupo, se devuelven los valores originales sin cambios.
    
    Input: recibe un grupo de filas/subset de un dataset, con o sin NANs. Objeto pandas Series, que representa una columna de datos
    Output: objeto pandas Series, donde los valores nulos han sido rellenados con la moda de la columna correspondiente.
    '''
    if len(x.dropna()) > 1:  # Verificar si hay más de un valor no nulo en el grupo
        return x.fillna(x.mode().iloc[0])  # Llenar los nulos con el modo
    else:
        return x  # Si solo tenemos un vino para calcular la moda, mantener los nulos



# Función para etiquetar y lematizar los tokens de una frase utilizando WordNet
def lemmatizer_get_wordnet_pos_phrase(frase_tokens):
    '''
    Esta función toma una lista de tokens de una frase como entrada y devuelve los tokens lematizados junto con sus etiquetas del "Part Of Speech" (POS).
    POS Se refiere a la categoría gramatical a la que pertenece una palabra en un contexto específico. Las etiquetas POS indican si una palabra es un sustantivo, 
    verbo, adjetivo, adverbio, etc. Estas etiquetas son esenciales para comprender la estructura y el significado de una oración en PNL.
    
    Input: Recibe una lista de tokens que representan una frase
    Output: devuelve dos listas: una lista con los tokens lematizados y otra lista con las etiquetas de las partes del discurso correspondientes a cada token.
    '''
    new_tokens = []
    pos_tokens = []

    tags = nltk.pos_tag(frase_tokens)
    wordnet_lemmatizer = WordNetLemmatizer() #cargamos el lematizador
    
    # Definir el diccionario de etiquetas para lematizar
    tag_dict = {"J": wordnet.ADJ,  # Adjetivo
            "N": wordnet.NOUN  # Nombre
           }
    
    for word, tag in tags:
        pos = tag_dict.get(tag[0], wordnet.NOUN)
        new_tokens.append(wordnet_lemmatizer.lemmatize(word, pos))
        pos_tokens.append(pos)
    
    return new_tokens, pos_tokens

    
    
    
# Función para normalizar, lematizar y eliminar stopwords de un texto
def normalizar_lemmatizar(texto):
    '''
    Esta función realiza varias operaciones en un texto dado:
    1. Reemplaza caracteres no alfabéticos por espacios y convierte todo el texto a minúsculas
    2. Tokeniza el texto en palabras: Utiliza nltk.word_tokenize(texto) para dividir el texto en palabras individuales.
    3. Lematiza los tokens y obtiene las etiquetas POS: solo nombres y adjetivos
    4. Filtra stopwords: Elimina las stopwords del castellano
    5. Une los tokens en un solo string y lo devuelve como salida de la función.
    
    Input: Recibe un texto en forma bruta/raw
    Output: devuelve el mismo texto normalizado, limpio de stopwords, tokenizado y lematizado listo para vectorizar
    '''
        
    # Cargar las stopwords y crear el lematizador una sola vez
    stopwords = set(nltk.corpus.stopwords.words('spanish'))
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Reemplazar caracteres no alfabéticos por espacios y convertir a minúsculas
    texto = re.sub("[^a-zA-ZñÑ]", " ", str(texto)).lower()
    # Tokenizar el texto en palabras
    tokens = nltk.word_tokenize(texto)
    # Lematizar los tokens y obtener las etiquetas POS
    tokens, _ = lemmatizer_get_wordnet_pos_phrase(tokens)
    # Filtrar stopwords
    tokens = [word for word in tokens if word not in stopwords]
    # Unir los tokens en un solo string
    return " ".join(tokens)
   

    
# Funcion que engloba todo el preprocesamiento del dataset de vinos
def preprocesamiento_bodeboca(dataset):
    '''
    Esta funcion carga un dataset con datos de vinos escrapeados de bodeboca y saca otro dataset limpio, preparado para meter en los modelos
    
    Input: dataset con datos de vino de bodeboca.com (dataset_raw)
    Output: dataset limpio despues de todo el preprocesamiento (df_vinos_limpio) 
    
    '''
       
    ##### 1. SELECCIONAMOS LAS COLUMNAS DE INTERESANTES
    # Lista de nombres de columnas que deseamos seleccionar
    columnas_deseadas = ['titulo', 'link', 'precio', 'rating', 'bodega', 'tipo',
                         'grado', 'anada', 'variedad', 'origen', 'vista', 'nariz', 'boca',
                         'temp_servir ', 'maridaje', 'clima', 'suelo', 'envejecimiento']
    # Hacemos el subset
    data = dataset.loc[:, columnas_deseadas]
    
    #### 2.LIMPIAMOS TEXTO DE COLUMNAS STRINGS DE SIMBOLOS RAROS - acentos principalmente
    data = data.applymap(limpiar_texto)
    data = data.applymap(limpiar_texto)
    
    #### 3. REVISAMOS COLUMNAS NUMERICAS:
    # Convertimos rating a float -- numerico 
    data["rating"]=data["rating"].str.replace(",",".")
    data["rating"]=data["rating"].astype(float)
    
    # Sacamos el grado de alcohol de la columna grado
    data["grado"]=data["grado"].str.replace("% vol.","")
    data["grado"]=data["grado"].astype(float)
    
    # Sacamos la temperatura de servir - cogemos la primera temperatura que aparece
    data['temp_servir '] = data['temp_servir '].str.extract(r'(\d{1,2})')
    data["temp_servir "]=data["temp_servir "].astype(float)
    
    #### 4. ELIMINAMOS DUPLICADOS
    data.drop_duplicates(inplace = True)
    
    #### 5. REDUCCION DE TIPOS DE VINO - creamos una nueva columna haciendo un reemplazo de tipos de vino para agrupar en menos tipos
    # Definir el diccionario de reemplazo
    diccionario_reemplazo = {"tinto": "tinto",
                             "red vermouth": "vermouth",
                             "blanco": "blanco",
                             "espumoso": "espumoso",
                             "amontillado": "generoso",
                             "tinto_reserva": "tinto",
                             "tinto reserva": "tinto",
                             'blanco fermentado en barrica': "blanco",
                             'white vermouth': "vermouth",
                             'manzanilla': "generoso",
                             'dulce px': "generoso",                        
                             'palo cortado': "generoso", 
                             'palo cortado vors': "generoso", 
                             'fino': "generoso",                         
                             'rosado' : "rosado",
                             'otro(s)': "generoso", 
                             'tinto joven': "tinto",
                             'tinto crianza': "tinto", 
                             'amontillado vors': "generoso",
                             "oloroso": "generoso",
                             'oloroso vors': "generoso",
                             'aromatised wine': "generoso", 
                             'blanco naturalmente dulce': "generoso",
                             'tinto dulce': "tinto",
                             'blanco dulce': "blanco", 
                             'orange wine': "blanco", 
                             'tinto gran reserva': "tinto", 
                             'cava': "espumoso",
                             'sweet moscatel': "generoso", 
                             'oloroso dulce': "generoso", 
                             'dulce px vors' : "generoso", 
                             'frizzante': "espumoso",
                             'rueda dorado': "blanco", 
                             'vermut dorado': "vermouth", 
                             'dulce': "generoso", 
                             'rancio': "generoso"                         
                            }
    # creacion de columna tipo2 con los nuevos tipos de vino (tinto, blanco, rosado, espumoso, generoso, vermouth)
    data['tipo2'] = data['tipo'].replace(diccionario_reemplazo)
    
    
    #### 6. REVISAMOS LA COLUMNA VARIEDAD - solo extraemos la variedad de uva sin los porcentajes
    data['variedad'] = data['variedad'].astype(str)
    data['variedad2'] = data['variedad'].apply(extraer_palabras)
    
    #### 7. REEMPLAZO DE NANs EN COLUMNAS DE TEXTO/STRING
    # Vamos a utilizar la columna `tipo` original para rellenar los NANs para que tenga sentido con el tipo de vino
    # Primero eliminamos las filas donde el tipo de vino es NA, sino nos da problemas
    data = data.dropna(subset=['tipo'])
    
    # Nulos de clima teniendo en cuenta por agrupacion de tipo de vino    
    data['clima'] = data.groupby('tipo')['clima'].transform(fillna_mode)
    
    # Nulos de suelo teniendo en cuenta por agrupacion de tipo de vino`    
    data['suelo'] = data.groupby('tipo')['suelo'].transform(fillna_mode)
    
    # Nulos de suelo teniendo en cuenta por agrupacion de tipo de vino 
    data['maridaje'] = data.groupby('tipo')['maridaje'].transform(fillna_mode)
    
    
    #### 8. Creamos columnas nuevas agrupan cata y caracteristicas fisicas de la viña y descripcion
    data["cata"] = data["vista"] + " " + data["nariz"] + " " + data["boca"]
    data ["fisico"] = data["clima"] + " " + data["suelo"]
    data["descripcion"] = data["cata"] + " " + data["maridaje"] 
    
    #### 8.1. REEMPLAZO DE NANs EN columnas cata, fisico y descripcion
    data['cata'] = data.groupby('tipo')['cata'].transform(fillna_mode)
    data['fisico'] = data.groupby('tipo')['fisico'].transform(fillna_mode)
    
    #### 9. Transformaciones de NLP - NORMALIZACION
    # trataremos las columnas: vista, nariz, boca, clima, suelo, maridaje
    # El flujo normal de NLP, sigue los siguientes pasos: Datos --> Normalizacion --> Vectorizacion --> aaplicar Modelo
    # Normalizacion: pasar a minusculas, tokenizar, eliminar stopwords, lematizacion ()   
    #Transformaciones de NLP por columna
    data["descripcion2"] = data.descripcion.apply(normalizar_lemmatizar)
    data["fisico2"] = data.fisico.apply(normalizar_lemmatizar)
    
    return data




###########################################################################################################
# cargamos el dataset de los vinos con datos `brutos` de bodeboca.com y las actualizaciones por parte de nosotros
dataset = pd.read_csv('df_vinos_raw.csv', encoding='latin1')

# aplicamos la funcion de preprocesamiento global
vinos = preprocesamiento_bodeboca(dataset)
vinos

# Guardar el DataFrame en un archivo para modelos CSV 
vinos.to_csv('df_vinos_clean.csv', index=False)