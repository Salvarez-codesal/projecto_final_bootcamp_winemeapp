# Proyecto final de bootcamp: Wine me App!
🔴 **Recomendador de vinos interactivo**

Este proyecto engloba los resultados del proyecto final del bootcamp en data science y machine learning de ID bootcamps. El objetivo principal es crear un motor de recomendacion de vinos basado en notas de cata y maridaje a partir de un vino dado.

![Banner](https://github.com/Salvarez-codesal/projecto_final_bootcamp_winemeapp/blob/main/imagenes/Banner.png)

_Motor de recomendacion de vinos que usa un modelo hibrido de NLP y TOPSIS_

## Tabla de contenido

* [Acerca del proyecto](#about-the-project)
* [Built With](#built-with)
* [Usage](#usage)
* [Autor](#authors)
* [Acknowledgements](#acknowledgements)

## Recycling
According to the [Waste Framework Directive](https://joint-research-centre.ec.europa.eu/scientific-activities-z/less-waste-more-value/definition-recycling_en), **recycling** is defined as

>  "any recovery operation by which waste materials are reprocessed into products, materials or substances whether for the original or other purposes"

### Beyond recycling
For a long time, we all became familiar with the 3R's (Recycling, Reduce, Reuse), but that concept is obsolete and more R's and other letters in the alphabet have joined the team in an effort to fight climate change, biodiversity loss, habitat fragmentation, desertification and much more.

- Refuse
- Reduce
- Reuse
- Repair
- Recycle
- Remake
- Restore
- Remember
- Respect

In websites like [Greenpeace](https://www.greenpeace.org.au/blog/beyond-reduce-reuse-recycle/) and [52ClimateActions](https://www.52climateactions.com/refuse-reduce-reuse-repair-recycle/full) you can find more information and ideas on how to lead a more sustaianle life.

## Objetivos principales

The aim of the project is to help people navigate the sometimes confussing world of recycling.

## ¿Como funciona? :camera:
1. Conecta con la web de WineMeApp: https://winemeup.streamlit.app/
2. Introduce el nombre de un vino que te guste y quieras recomendaciones de vinos similares
3. Confirma si ese vino está en nuestra base de datos y se mostrara una lista de todos los vinos que contienen ese nombre
4. Introduce el nombre del vino exactamente como aparece en nuestra base de datos

- Resultados modelo NLP:
6. Se genera la primera lista de recomendación de vinos basada en notas de catas y maridaje

- Resultados del modelo TOPSIS:
7. Se pide al usuario un rango de precios
8. Se realiza la segunda recomendacion, filtrando por ese rango de precio y estableciendo una relacion ponderada entre el "precio" y el "ranking"
9. Se muestra un gráfico con esa relación (el tamaño de los puntos refleja el ranking, el color, ,el tipo de vino)
