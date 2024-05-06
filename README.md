# Proyecto final de bootcamp: Wine me App!
🔴 **Recomendador de vinos interactivo**

Este proyecto engloba los resultados del proyecto final del bootcamp en data science y machine learning de ID bootcamps. El objetivo principal es crear un motor de recomendacion de vinos basado en notas de cata y maridaje a partir de un vino dado.

![Banner](https://github.com/Salvarez-codesal/projecto_final_bootcamp_winemeapp/blob/main/imagenes/Banner.png)

_Motor de recomendacion de vinos que usa un modelo hibrido de NLP (Natural Language Processing) y TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)_


## Objetivos principales

Este proyecto tiene como objetivo desarrollar un motor de recomendación de vinos que aprovecha las notas de cata y maridaje. La verdadera innovación radica en la capacidad del cliente para personalizar las recomendaciones, permitiéndole ajustar la importancia relativa del precio y el rating de los vinos. Este enfoque único, de modelo hibrido, garantiza una experiencia de selección de vinos completamente adaptada a las preferencias individuales del usuario.

Aquí podemos ver la distribución de los clusters de todos los vinos de nuestra base de datos en base a notas de cata y maridaje que los explican:

![cluster_vinos_NLP](https://github.com/Salvarez-codesal/projecto_final_bootcamp_winemeapp/blob/main/imagenes/cluster_vinos_NLP.PNG)


De esta manera, cada vez que un usuario seleccione un vino de los 4792 de la base de datos (cada uno de los puntos del grafico), el motor de recomendacion buscara aquellos mas afines semanticamente en notas de cata y maridaje, es decir aquellos que esten mas proximos a ellos en el grafico de clusters. Posteriormente, al incorporar el modelo TOPSIS, permitimos que el usuario tenga mayor poder de decision, permitiendole que le de los pesos o importancia a los precios de los vinos y sus valoraciones por otros usuarios y expertos (ratings)


## Conclusiones del proyecto:

🟢 **Innovación Personalizada**: Desarrollamos un motor de recomendación de vinos único que permite a los usuarios ajustar sus preferencias tanto de caracteristicas del vino como de precio y rating, ofreciendo una experiencia adaptada a cada individuo.

🟢 **Amplia Variedad**: Nuestro enfoque basado en notas de cata y maridaje garantiza recomendaciones diversas, rompiendo con los prejuicios y ofreciendo resultados sorprendentes para los usuarios.

🟢 **Éxito en la Implementación**: Logramos integrar eficazmente nuestro modelo híbrido en una aplicación web interactiva, proporcionando funcionalidades completas y visualizaciones intuitivas.


## Futuras líneas de investigación:

🟨 **Visualización Geográfica**: Explorar la ubicación de las bodegas de vinos recomendados en un mapa para proporcionar una perspectiva geográfica de las recomendaciones.

🟨 **Ampliación del Modelo TOPSIS**: Extender el modelo TOPSIS a más categorías para mejorar la capacidad de personalización y decisión del usuario.

🟨 **Exploración de Datos Avanzada**: Analizar en profundidad los datos de los vinos para crear nuevas características numéricas y gráficos de radar que enriquezcan la experiencia del usuario.

🟨 **Inclusión de Vinos Internacionales**: Integrar un segundo modelo NLP para vinos internacionales basado en notas de cata, utilizando conjuntos de datos en inglés y explorando la traducción automática.


## ¿Como funciona WineMeApp?: 🌐🍷
1. Conecta con la web de WineMeApp: https://winemeup.streamlit.app/
2. Introduce el nombre de un vino que te guste y quieras recomendaciones de vinos similares
3. Confirma si ese vino está en nuestra base de datos y se mostrara una lista de todos los vinos que contienen ese nombre
4. Introduce el nombre del vino exactamente como aparece en nuestra base de datos

- **Resultados modelo NLP**:
6. Se genera la primera lista de recomendación de vinos basada en notas de catas y maridaje

- **Resultados del modelo TOPSIS**:
7. Se pide al usuario un rango de precios
8. Se realiza la segunda recomendacion, filtrando por ese rango de precio y estableciendo una relacion ponderada entre el "precio" y el "ranking"
9. Se muestra un gráfico con esa relación (el tamaño de los puntos refleja el ranking, el color, ,el tipo de vino)
