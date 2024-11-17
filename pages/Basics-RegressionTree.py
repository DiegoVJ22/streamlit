import streamlit as st
from PIL import Image

# Título principal
st.title("Fundamentos de Árboles de Regresión")

# Introducción
st.header("¿Qué es un árbol de decisión?")
st.write(
    """Un árbol de regresión es un tipo de árbol de decisión que se utiliza para predecir variables de respuesta continuas. Es un algoritmo sencillo y rápido que suele utilizarse para predecir datos de muestra categóricos, discretos o no lineales en computer science. Son un conjunto de reglas sucesivas que ayudan a tomar una decisión."""
)

arbol_decision = Image.open("static/regressionTree/fit_unfit.png")
st.image(arbol_decision, caption="Ejemplo visual de un árbol de decisión.")

# Tipos de árboles
st.header("Tipos de árboles de decisión")
st.write(
    """Existen dos tipos principales de árboles de decisión:

    - Árboles de regresión: Predicen una variable continua.
    - Árboles de clasificación: Predicen una variable categórica.
    """
)

# Sección: Gráfica comparativa
st.header("Comparación entre árboles de clasificación y regresión")
st.write(
    "La siguiente gráfica compara un árbol de clasificación y un árbol de regresión."
)
image_comparativa = Image.open("static/regressionTree/tipos_arboles.png")
st.image(image_comparativa, caption="Comparación entre árboles de clasificación y regresión")

st.header("Árboles de regresión")
st.markdown(
    r"""
    Un árbol de regresión consiste en hacer preguntas de tipo $x_k \leq c$ para cada una de las covariables, 
    de esta forma el espacio de las covariables es dividido en hiper-rectángulos y todas las observaciones 
    que queden dentro de un hiper-rectángulo tendrán el mismo valor estimado $\hat{y}$. La idea es que 
    la predicción sea un valor numérico en lugar de una categoría. La partición del espacio se hace de manera 
    repetitiva para encontrar las variables y los valores de corte $c$ de tal manera que se minimice la 
    función de costos:
    """,
    unsafe_allow_html=False,
)

st.latex(r"""
\sum_{i=1}^{n} (y_i - \hat{y}_i)^2
""")

st.write(
    """Mientras que en los árboles de clasificación se utiliza la entropía o la impureza de Gini para medir la homogeneidad de un nodo, en los árboles de regresión se utiliza como métrica la desviación estándar (σ) de la variable respuesta.
    """
)
st.latex(r"""
\sigma_X = \sum_{r \in X} P(r) \cdot \sigma_r
""")
st.write(
    """
    El objetivo es minimizar la dispersión en cada grupo después de dividir. La mejor división será aquella que deje los valores de cada grupo lo más parecidos posible, es decir, con la menor desviación estándar ponderada.
    """
)
image_particiones = Image.open("static/regressionTree/ilustracion_arb_regresion.png")
st.image(image_particiones, caption="Particiones del espacio y estructura del árbol.")

# Importancia de la desviación estándar
st.header("¿Por qué se utiliza la desviación estándar?")
st.write(
    """La desviación estándar ponderada se utiliza en árboles de regresión para evaluar 
    la calidad de las divisiones. El objetivo es minimizar la dispersión dentro de 
    cada región, logrando grupos más homogéneos."""
)

# Visualizaciones adicionales
# st.header("Ejemplo visual de decisiones en un árbol")
# st.write(
#     "El siguiente diagrama muestra un ejemplo simplificado de decisiones tomadas en un árbol para predecir el estado de salud." 
# )
# image_decision = Image.open("/mnt/data/image.png")
# st.image(image_decision, caption="Ejemplo de decisiones en un árbol de regresión")

# Conclusiones
st.header("Conclusiones")
st.write(
    """Los árboles de regresión son herramientas poderosas y flexibles que dividen 
    los datos en regiones manejables, optimizando las predicciones para variables continuas. 
    Su sencillez e interpretabilidad los convierten en una opción popular en machine learning."""
)

st.write("Explora los conceptos clave y las visualizaciones proporcionadas para entender mejor su funcionamiento.")