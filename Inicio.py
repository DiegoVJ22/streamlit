import streamlit as st

st.set_page_config(page_title="Regresión Logística")
st.title("Regresión Logística")

st.write("""
Es un método estadístico utilizado para modelar problemas de clasificación binaria, donde el resultado es una de dos posibles categorías, como *sí* o *no*, *verdadero* o *falso*. A diferencia de la regresión lineal, que predice valores continuos, la regresión logística estima la probabilidad de que una instancia pertenezca a una clase determinada.
""")

# Aquí puedes insertar una imagen que ilustre un ejemplo de clasificación binaria, como separar puntos en un plano según dos categorías.

# Fundamentos Matemáticos
st.header("Fundamentos Matemáticos")

st.subheader("La Función Sigmoide")

st.write("""
El núcleo de la regresión logística es la **función sigmoide** o logística, que transforma cualquier valor real en un valor entre 0 y 1, interpretado como una probabilidad.
""")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write('')  # Columna vacía

with col2:
    st.image('static/regresion_logistica/sigmoide1.png')
    st.write("""
*donde \( z \) es una combinación lineal de las variables independientes.*
""")
with col3:
    st.write('')  # Columna vacía

st.write("Si representa esta ecuación de regresión logística, obtendrá una curva en S como la que se muestra a continuación.")
st.image('static/regresion_logistica/grafica-sigmoide.png')
st.write("Devuelve solo valores entre 0 y 1 para la variable dependiente, al margen de los valores de la variable independiente. Así es como la regresión logística estima el valor de la variable dependiente.")


# Aquí debes incluir la fórmula anterior y, si es posible, un gráfico de la función sigmoide mostrando cómo transforma valores desde el eje real al intervalo (0,1).

st.subheader("Modelo de Regresión Logística")

st.write("""
El modelo matemático se expresa como:

""")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/modelo.png")
with col3:
    st.write('')



st.write("""
*donde:*

- P(Y=1∣X) es la probabilidad de que el resultado sea la clase 1 dado el vector de características X.
- β0 es el término intercepto.
- β1 son los coeficientes del modelo.
""")

# Aquí puedes mostrar la fórmula anterior y explicar cada componente. También sería útil incluir un diagrama que represente cómo las variables independientes se combinan linealmente y se transforman mediante la función sigmoide.

# Interpretación de los Coeficientes
st.header("Interpretación de los Coeficientes")

st.write("""
Los coeficientes βi indican la influencia de cada variable independiente en la probabilidad del resultado. Un coeficiente positivo significa que a medida que la variable aumenta, también lo hace la probabilidad de que Y sea 1.
""")

# Estimación de Parámetros
st.header("Estimación de Parámetros")

st.write("""
Para encontrar los valores óptimos de β, se utiliza el método de **Máxima Verosimilitud**, que busca maximizar la probabilidad de observar los datos dados los parámetros del modelo.
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/max_verosimi.png")
with col3:
    st.write('')

st.write("Se busca maximizar 𝐿(𝜃), lo que significa encontrar los valores de 𝜃 que hacen más probable la observación de los datos")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/maxverosim1.png")
with col3:
    st.write('')

st.write("Muestra cómo cambia 𝐿(𝜃) en función de los valores posibles de 𝜃. El pico representa el valor óptimo de 𝜃, donde la probabilidad de observar los datos es máxima.")



# Aquí puedes incluir la función de verosimilitud y explicar brevemente el proceso de optimización.

# Registrar Probabilidades
st.header("Registrar Probabilidades")

st.write("""
El modelo logit también puede determinar la relación entre el éxito y el fracaso o registrar las probabilidades.
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/probabilidad.png")
    st.write("""
*Aquí, p representa la probabilidad de éxito, y 1-p la probabilidad de fracaso.*
""")
with col3:
    st.write('')

st.write("Esta función transforma valores de probabilidades (entre 0 y 1) en valores que pueden ir desde −∞ hasta +∞. Esto permite que el modelo lineal ajuste mejor los datos en contextos probabilísticos")



st.write("Muestra cómo la función logit mapea probabilidades pequeñas (p cerca de 0) en valores negativos grandes y probabilidades altas (p cerca de 1) en valores positivos grandes. La función es simétrica respecto al punto medio (p = 0.5).")

# Tipos de Regresión Logística
st.header("Tipos de Regresión Logística")

st.write("""
Hay tres enfoques para el análisis de regresión logística basados en los resultados de la variable dependiente.
""")

st.subheader("Regresión Logística Binaria")

st.write("""
Funciona bien para problemas de clasificación binaria que solo tienen dos resultados posibles. La variable dependiente solo puede tener dos valores, como sí y no o 0 y 1.
""")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/regresion_binaria.png")
with col3:
    st.write('')

st.write("Cada gráfico muestra cómo el modelo clasifica puntos en dos categorías (rojo y azul) con diferentes niveles de ajuste o complejidad.")

st.subheader("Regresión Logística Multinomial")

st.write("""
Puede analizar problemas que tienen varios resultados posibles, siempre y cuando el número de resultados sea finito. funciona mapeando los valores de resultado con diferentes valores entre 0 y 1. Como la función logística puede devolver un rango de datos continuos, como 0,1, 0,11, 0,12, etc., la regresión multinomial también agrupa el resultado a los valores más cercanos posibles.
""")

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/formula_regresion_multi.jpg")
with col3:
    st.write('')

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/regresion_multinomial.png")
with col3:
    st.write('')

st.write("Muestra cómo el modelo divide el espacio en regiones, asignando cada punto a una clase (colores diferentes) según su probabilidad calculada.")

# st.subheader("Regresión Logística Ordinal")

# st.write("""
# Es un tipo especial de regresión multinomial para problemas en los que los números representan rangos en lugar de valores reales. Por ejemplo, puede utilizar la regresión ordinal para predecir la respuesta a una pregunta de una encuesta en la que se pide a los clientes que clasifiquen su servicio como malo, regular, bueno o excelente en función de un valor numérico, como el número de artículos que le han comprado a lo largo del año.
# """)

# Evaluación del Modelo
st.header("Evaluación del Modelo")

st.write("""
Es fundamental evaluar el desempeño del modelo utilizando métricas adecuadas:
""")

st.subheader("Matriz de Confusión")

st.write("""
Resume las predicciones correctas e incorrectas en cada clase.
""")

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.write('')

with col2:
    st.image("static/regresion_logistica/matriz_confusion.png")
with col3:
    st.write('')

st.write("Cada columna de la matriz representa el número de predicciones de cada clase, mientras que cada fila representa a las instancias en la clase real., o sea en términos prácticos nos permite ver  qué tipos de aciertos y errores está teniendo nuestro modelo a la hora de pasar por el proceso de aprendizaje con los datos.")

# Incluir una tabla que represente una matriz de confusión.

st.subheader("Exactitud, Precisión, Recall y F1-Score")

st.subheader("Exactitud (Accuracy)")
st.write("Es la proporción de predicciones totales que el modelo acertó; mide con qué frecuencia el modelo clasifica correctamente tanto casos positivos como negativos.")

st.subheader("Precisión")
st.write("Es la proporción de predicciones positivas que realmente son correctas; indica la exactitud de las predicciones positivas del modelo.")

st.subheader("Recall")
st.write("Es la proporción de casos positivos reales que el modelo identificó correctamente; refleja la capacidad del modelo para encontrar todos los casos positivos.")

st.subheader("F1-Score")
st.write("Es la media armónica entre precisión y recall; proporciona un equilibrio entre ambos, especialmente útil cuando existe un desequilibrio entre clases.")

st.subheader("Fórmulas")
col1, col2 = st.columns([1, 1])
with col1:
    st.image("static/regresion_logistica/formulas_metricas.png")

with col2:
    st.write("""
    *donde:*

    - TP: True Positive, son instancias positivas correctamente clasificadas.
    - TN: True Negative, son instancias negativas correctamente clasificadas
    - FP: False Positive, son instancias negativas incorrectamente clasificadas como positivas
    - FP: False Negative, son instancias positivas incorrectamente clasificadas como negativas.

    """)

# Aquí puedes mostrar las fórmulas de estas métricas y, si es posible, gráficos que ayuden a ilustrar su interpretación.

# Ventajas y Limitaciones
st.header("Ventajas y Limitaciones")

st.subheader("Ventajas")

st.write("""
- Simple de implementar y interpretar.
- Eficiente para problemas de clasificación binaria.
- Probabilidades de salida interpretables.
""")

st.subheader("Limitaciones")

st.write("""
- No maneja bien relaciones no lineales a menos que se incorporen términos polinómicos o interacciones.
- Sensible a outliers.
""")

# Aplicaciones de la Regresión Logística
st.header("Aplicaciones de la Regresión Logística")

st.write("""
La regresión logística es ampliamente utilizada en diversos campos:
""")

st.write("""
- **Medicina**: Predicción de la presencia o ausencia de una enfermedad.
- **Finanzas**: Evaluación de la probabilidad de incumplimiento de crédito.
- **Marketing**: Determinar si un cliente realizará una compra.
- **Seguridad**: Detección de spam o correos electrónicos fraudulentos.
""")

st.subheader("Regresión logística vs. regresión lineal")

st.write("La regresión lineal predice una variable dependiente continua mediante el uso de un conjunto dado de variables independientes. Una variable continua puede tener un rango de valores, como el precio o la antigüedad. Por lo tanto, la regresión lineal puede predecir los valores reales de la variable dependiente. Puede responder a preguntas como ¿Cuál será el precio del arroz después de 10 años? A diferencia de la regresión lineal, la regresión logística es un algoritmo de clasificación. No puede predecir los valores reales de los datos continuos. Puede responder a preguntas como ¿Aumentará el precio del arroz un 50 % en 10 años?")
# Incluir ejemplos gráficos o casos de uso específicos puede ayudar a ilustrar estas aplicaciones.