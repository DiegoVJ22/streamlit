import streamlit as st

st.set_page_config(page_title="Machine Learning App")
st.title("Machine Learning App 🤖")

# Procesamiento de Datos
with st.container():
    st.subheader("Procesamiento de Datos 📊")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Este módulo se encarga de preparar y limpiar los datos antes de ser utilizados en el modelo. 
        Realiza tareas de manejo de valores faltantes, codificación de variables categóricas y escalado 
        de características. El procesamiento adecuado de los datos asegura que el modelo tenga un buen 
        rendimiento y precise de predicciones confiables.
        """)
    with col2:
        st.image("static/procesamiento.webp", use_container_width=True)

# Regresión Lineal Simple
with st.container():
    st.subheader("Regresión Lineal Simple 📈")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Este modelo predice una variable objetivo a partir de una sola variable independiente. 
        La regresión lineal simple es útil para entender relaciones lineales directas y se visualiza fácilmente 
        a través de gráficos de dispersión con una línea de tendencia. Este módulo permite entrenar el modelo 
        y evaluar su precisión.
        """)
    with col2:
        st.image("static/lineal_simple.webp", use_container_width=True)

# Regresión Lineal Múltiple
with st.container():
    st.subheader("Regresión Lineal Múltiple 📈📊")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        La regresión lineal múltiple permite analizar el impacto de múltiples variables independientes en 
        una variable dependiente. Este módulo incluye técnicas avanzadas como la eliminación hacia atrás, 
        que optimiza el modelo seleccionando solo las variables más relevantes para hacer predicciones 
        más precisas.
        """)
    with col2:
        st.image("static/lineal_multiple.webp", use_container_width=True)
    
"""
1. Regresión Logística
Ventajas:

Es simple y eficiente para problemas lineales.
Produce probabilidades, lo que permite interpretar la confianza en las predicciones.
Es menos propenso al sobreajuste en conjuntos de datos pequeños.
Ideal para problemas donde las variables independientes tienen una relación lineal con la variable dependiente.
Desventajas:

No maneja bien relaciones no lineales entre las variables.
Requiere que las características estén correctamente escaladas para un rendimiento óptimo.
Cuándo usarla:

Cuando el problema tiene una estructura lineal o cuasi-lineal.
Cuando interpretabilidad y simplicidad son importantes.
2. Árbol de Clasificación (CART)
Ventajas:
Captura relaciones no lineales y de interacción entre variables.
No requiere escalar los datos o aplicar transformaciones.
Fácil de interpretar gráficamente como un árbol.
Funciona bien con datos categóricos y mixtos (numéricos y categóricos).
Desventajas:
Propenso al sobreajuste, especialmente si el árbol crece demasiado (puede mitigarse con técnicas como la poda).
Menor robustez a pequeñas variaciones en los datos (puede cambiar drásticamente si los datos cambian ligeramente).
Cuándo usarlo:
Cuando hay relaciones no lineales o interacciones complejas entre variables.
Cuando se requiere una representación visual de las reglas de decisión.
Criterios de Comparación
Desempeño (Matriz de Confusión y Métricas):

Si los datos tienen una estructura lineal, Regresión Logística tiende a tener mejor precisión y recall.
Si las relaciones entre variables son no lineales, CART suele capturar mejor la complejidad, mejorando la F1-Score.
Interpretabilidad:

Regresión Logística es más interpretable para modelos matemáticos.
CART es intuitivo y fácil de entender para no expertos.
Robustez frente a Datos:

CART funciona mejor con datos faltantes o categorías.
Regresión Logística requiere un preprocesamiento más exhaustivo.
Rendimiento en Datos Desequilibrados:

Regresión Logística puede ser afectada por datos desbalanceados, pero se puede ajustar con pesos.
CART tiende a sesgarse hacia la clase mayoritaria, pero técnicas como el balanceo pueden ayudar.
"""
