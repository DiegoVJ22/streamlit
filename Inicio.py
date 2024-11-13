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
        st.image("static/procesamiento.webp", use_column_width=True)

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
        st.image("static/lineal_simple.webp", use_column_width=True)

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
        st.image("static/lineal_multiple.webp", use_column_width=True)
