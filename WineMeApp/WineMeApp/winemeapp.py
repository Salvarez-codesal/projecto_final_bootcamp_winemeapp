import streamlit as st
from streamlit_option_menu import option_menu
from funciones_sistema_recomendacion_vinos import *
import requests

# data = "WineMeApp/df_vinos_modelos.csv"
data = pd.read_csv('WineMeApp/df_vinos_modelos.csv')


# Tab Info
st.set_page_config(
	page_title = "WineMeApp!",
    page_icon=":wine_glass:",
    layout="wide")
# ------------------------------------------------

# Importar imagen desde GDrive
file_id = "1FUeYXfNwHDSxVzn3HsctD8b7oQU4fXPz"
url = f"https://drive.google.com/uc?export=view&id={file_id}"
response = requests.get(url)
st.image(response.content)

# -----------------------


selected = option_menu(
	menu_title = None,
	options = ["Inicio", "WineMeApp!", "Recursos", "Quiénes somos"],
	icons = ['house-door', "menu-button-wide-fill","journals", "person lines fill"],
	menu_icon = "",
	orientation = "horizontal",
	styles={
    "container": {"padding": "0!important", "background-color": "#FFFFFF"},
    "icon": {"color": "black", "font-size": "30px"},
    "nav-link": {"font-size": "20px", "text-align": "left","font-family": 'Cooper Black',"color": "black", "margin":"0px", "--hover-color": "#0000"},
    "nav-link-selected": {"background-color": "#e51133"},
})
if selected =="Inicio":
    st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FFFFF;} 
            </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Introducción</p>', unsafe_allow_html=True)
    col1, col2, col3= st.columns([0.5, 0.3, 0.2])
    with col1:  # To display the header text using css style
        st.write("¡Te presento un motor de recomendación de vinos que va más allá de lo convencional!")
        st.write(
            "Olvídate de las sugerencias basadas solo en el tipo de vino. Nuestro sistema innovador analiza tus preferencias y necesidades,"
            " utilizando detalladas notas de cata y maridaje para encontrar la opción perfecta para cada ocasión.")
        st.markdown(""" <style> .font {
                    font-size:35px ; font-family: 'Cooper Black'; color: #FFFFF;} 
                    </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">¿Qué nos hace diferentes?</p>', unsafe_allow_html=True)
        st.markdown("* **Recomendaciones personalizadas:** Ajusta la importancia del precio y la puntuación para encontrar vinos que se adapten a tu presupuesto y gusto.")
        st.markdown("* **Más allá de los estereotipos:** No te limitamos a un solo tipo de vino. Descubre nuevos favoritos, desde tintos suaves hasta blancos frescos o espumosos.")
        st.markdown("* **Datos selectos por expertos:** Nuestras recomendaciones se basan en extensas notas de cata y maridaje, lo que te garantiza encontrar vinos que realmente disfrutarás.")
        st.markdown("* **Di adiós a las experiencias de vino promedio.** Deja que nuestro motor te guíe hacia nuevos descubrimientos y momentos inolvidables.")
        st.markdown("#####  ¿Quieres saber más?")
    with col2:
        st.write("")
        # Importar imagen desde GDrive
        file_id = "1ncIfBeSzk-BaKjU2gaM037-L_kCF8_cf"
        url = f"https://drive.google.com/uc?export=view&id={file_id}"
        response = requests.get(url)
        st.image(response.content)

    st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FFFFF;} 
            </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Motivación</p>', unsafe_allow_html=True)
    st.write("Muchos de nosotros nos hemos encontrado en la incómoda situación de no saber qué vino pedir en un restaurante o al comprar en una tienda especializada. A menudo, terminamos"
             " optando por lo mismo de siempre o por lo que pide alguien más, sin explorar la amplia gama de opciones disponibles.")
    st.write(
        'Es en estos momentos cuando un sistema de recomendación de vinos, ya sea en la sección de vinos de grandes almacenes o supermercados gourmet, o bien cuando vas a pedir'
        ' las bebidas en un restaurante, sería de gran utilidad. ¡Y es precisamente aquí donde nuestro recomendador <span style="color: #E51133; font-weight: bold;"> WineMeApp!</span> entra en juego!', unsafe_allow_html=True)

    st.write("Como parte del equipo, durante nuestra experiencia como expatriados en el Reino Unido, notamos que los vinos blancos de Chardonnay eran una opción particularmente recurrente. Si bien esta variedad es popular, "
             "nos sorprendió que no se exploraran las diversas opciones que ofrece el mundo del vino, incluyendo vinos españoles de regiones como Jerez de Frontera (Palomino, Pedro Ximénez, Tintilla de Rota) o Zamora"
             " (Malvasía Castellana, Moscatel de Grano Menudo, Verdejo).")
    st.write(
        'Es por ello que hemos desarrollado <span style="color: #E51133; font-weight: bold;"> WineMeApp!</span>, un motor de recomendación que te ayuda a elegir el vino perfecto para cada ocasión. Al considerar tus preferencias,'
        ' el tipo de comida y la ocasión, <span style="color: #E51133; font-weight: bold;"> WineMeApp!</span> te ofrece'
        ' recomendaciones personalizadas para que puedas descubrir nuevos vinos y disfrutar al máximo de cada experiencia.', unsafe_allow_html=True)

    st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FFFFF;} 
            </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Objetivo</p>', unsafe_allow_html=True)
    st.write("El proyecto utiliza un modelo híbrido de machine learning para recomendar vinos basados en las preferencias del usuario respecto a las notas de cata y maridaje. El usuario introduce el "
             "nombre del vino, verifica su disponibilidad en la base de datos y luego establece un rango de precio y la importancia del precio y el rating del vino para obtener recomendaciones personalizadas.")


# -----------------------------------------------------------------------------

if selected =="WineMeApp!":
    st.markdown("# Modelo de recomendación")
    st.write("El modelo de recomendación creado en este proyecto es un modelo "
             "híbrido que combina dos enfoques poderosos")
    st.markdown("## :wine_glass: Modelo 1: Recomendador en base a Cata y Maridaje")
    st.write("En primer lugar implementamos un sistema de recomendación basado en notas de cata"
             " y maridaje mediante Procesamiento del Lenguaje Natural (NLP).")
    st.write("Al introducir un vino para buscar recomendaciones similares, el sistema busca vinos con"
             " mayor similitud semántica, aplicando similitud del coseno, en las notas de cata y maridaje,"
             " proporcionando un ranking de los 10 vinos más afines.")

    # ---- BUSQUEDA EN LA LISTA DE 4K VINOS ----
    st.markdown("#### Paso 1: Comprueba si tu vino favorito está en nuestra base de datos")
    texto_busqueda = st.text_input(' ')
    if st.button("Encuentra tu vino 🔎 "):  # Si se pulsa el botón
        st.markdown(" ##### Resultados encontrados en nuestra base de datos: ")
        st.dataframe(buscar_vino(data, texto_busqueda, embeddings2))

    # ----- RECOMENDATOR ----
    st.markdown("#### Paso 2: Introduce el nombre del vino para el cual deseas buscar recomendación")
    nombre_vino = st.text_input("")
    if st.button("Recomendador 🍀 "):  # Si se pulsa el botón
        st.markdown(f" ##### Estos son los vinos que te recomendamos para *{nombre_vino}* ")
    recomendaciones = recomendar_vino(data, nombre_vino, embeddings2)
    st.dataframe(recomendaciones)

    # ---- TOPSIS ----
    st.markdown("##  :wine_glass::wine_glass: Modelo 2: TOPSIS")
    st.write("La versión actual de WineMeApp! tiene definidos por defecto un filtrado de los vinos con"
             " precios comprendidos entre 5 y 150 euros  y de los pesos otorgados a las variables precio (80%) y rating (20%)  ")
    st.markdown("##### Paso 3: Comprueba tu recomendación con el modelo TOPSIS*")

    vinos = filtrar_por_precio(recomendaciones, precio_min, precio_max)

    if st.button("Topsis :tada:"):  # Si se pulsa el botón
        st.write(topsis_general(recomendaciones, nombre_vino))
    # ------



# -----------------------------------------------------------------------------
if selected =="Recursos":
    col1, col2, col3, col4, col5 = st.columns([0.20, 0.20, 0.20, 0.20, 0.20])
    with col1:
        st.markdown("##  Páginas Web ")
        st.markdown(" * **Vinetur** ")
        st.markdown(" * **Vinissimus** ")
        st.markdown(" * **Bodeboca** ")
        st.markdown(" * **OEMV** ")
    with col2:
        st.header(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("""[www.vinetur.com](https://www.vinetur.com/)""")
        st.write("""[www.vinissimus.com/](https://www.vinissimus.com/es/)""")
        st.write("""[www.bodeboca.com](https://www.bodeboca.com/)""")
        st.write("""[www.oemv.es](https://www.oemv.es/)""")

    with col3:
        st.markdown("##  Librerías  ")
        st.markdown(" * Pandas")
        st.markdown(" * Numpy")
        st.markdown(" * Seaborn")
        st.markdown(" * Matplotlib")
    with col4:
        st.markdown(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown(" * Sklearn")
        st.markdown(" * TensorFlow")
        st.markdown(" * Plotly")
        st.markdown(" * Scipy")
    st.write("----")

# -----------------------------------------------------------------------------
if selected == "Quiénes somos":
    col1, col2, col3 = st.columns([0.3, 0.3, 0.3])
    with col1:  # To display the header text using css style
        with st.container():
            # Importar imagen desde GDrive
            file_maria_id = "1sCRq07Z-kYc8bGhZOAtBj_UyqSe7ZogQ"
            url_pic_maria = f"https://drive.google.com/uc?export=view&id={file_maria_id}"
            response_maria = requests.get(url_pic_maria)
            st.image(response_maria.content)
            st.markdown(
                """ #### María Pérez Sebastián :computer:""")
            st.write("""
            Actualmente estoy dando un giro a mi carrera profesional formándome como Data Scientist. 
            """)
            st.write("""
            Anteriormente, me he dedicado casi 10 años al desarrollo integral de proyectos arquitectónicos. Me considero una persona 
            responsable, comprometida y con una alta capacidad de trabajo. Me gusta trabajar en equipo, aportar ideas personales y aprender de la 
            forma de trabajar de los otros miembros positivamente.
            """)
            st.write("""[Github](https://github.com/marpezseb),
                 [LinkedIn](https://www.linkedin.com/in/-mps2024/) """)
    with col2:
        # Importar imagen desde GDrive

        file_ivan_id = "1s0koO8Ug2J-7nhTvdMQTIKCZZpfaMfq6"
        url_pic_ivan = f"https://drive.google.com/uc?export=view&id={file_ivan_id}"
        response_ivan = requests.get(url_pic_ivan)
        st.image(response_ivan.content)
        st.markdown(
            """
            #### Iván Pinto Grilo :computer:
            """)
        st.write(""" Curioso por naturaleza, data rookie y mente inquieta.""")
        st.write("""He decidido redirigir mi carrera hacia la ciencia de datos a través de un Bootcamp en Data Science y Machine Learning. 
        Con experiencia internacional en puestos de responsabilidad donde he podido ver la importancia de los datos,
         .""")
        st.write(""" Siempre con algun proyecto en mente""")
        st.write(""" [Github ](https://github.com/ivanpgdata), 
            [LinkedIn](https://www.linkedin.com/in/ivanpgdata/) """)
    with col3:
        # Importar imagen desde GDrive
        file_soraya_id = "1rzJvRfXgB61WgJloyBSYFRtUnirBW2j8"
        url_pic_soraya = f"https://drive.google.com/uc?export=view&id={file_soraya_id}"
        response_soraya = requests.get(url_pic_soraya)
        st.image(response_soraya.content)
        st.markdown(
            """
            #### Soraya Alvarez Codesal  :computer:""")
        st.write("""
        Científica y analista de Datos con más de 10 años de experiencia en investigación y
         consultoría multidisciplinar a nivel internacional, trabajando y formándome en 6 países distintos.""")
        st.write(""" Me encanta investigar y desarrollar nuevos conocimientos. Por ello, me ilusiona poder seguir creciendo profesionalmente en el 
        sector tecnológico, para generar un impacto positivo en la vida de las personas y las empresas a través de la búsqueda de soluciones y decisiones
         mas sosteniblesbasadas en datos.""")
        st.write("""[Github](https://github.com/Salvarez-codesal),
             [LinkedIn](https://www.linkedin.com/in/sorayaalvarezcodesal/) """)

    st.write("-------")
    frase = "#### “El que sabe degustar no bebe demasiado vino, pero disfruta sus suaves secretos.” ― Salvador Dalí."
    st.markdown(frase)
