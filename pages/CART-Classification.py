import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="CART Clasificación")
st.title("Árbol de Decisión - Clasificación 🌳")

# Definir nombres de las clases
class_names = ["Vino Tinto", "Vino Blanco", "Vino Rosado"]

# Cargar los datos desde un archivo CSV
df = pd.read_csv("csv/vino-dataset.csv")
X = df.iloc[:, :-1]  # Características
y = df.iloc[:, -1]   # Clase objetivo

# Mostrar toda la tabla de datos cargados
st.subheader("Datos Cargados:")
st.dataframe(df, height=500)  # Altura ajustable para desplazar la tabla

# Dividir en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el modelo de Árbol de Decisión
modelo = DecisionTreeClassifier(criterion="gini", random_state=42)
modelo.fit(x_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(x_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Mostrar resultados
st.subheader("Resultados del Modelo:")
st.write(f"**Precisión:** {accuracy:.2f}")
st.write("**Matriz de Confusión:**")
st.write(conf_matrix)

# Visualización de la matriz de confusión
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax1)
ax1.set_ylabel("Etiqueta Real")
ax1.set_xlabel("Etiqueta Predicha")
ax1.set_title("Matriz de Confusión")
st.pyplot(fig1)

# Visualizar el Árbol de Decisión
st.subheader("Visualización del Árbol de Decisión:")
fig2, ax2 = plt.subplots(figsize=(12, 8))
tree.plot_tree(modelo, feature_names=X.columns, class_names=class_names, filled=True, ax=ax2)
st.pyplot(fig2)

# Importancia de características
importancias = modelo.feature_importances_
st.subheader("Importancia de las Características:")
st.dataframe(pd.DataFrame({"Características": X.columns, "Importancia": importancias}).sort_values(by="Importancia", ascending=False))

# Entrada de datos personalizada en el sidebar
st.sidebar.header("Predicción Personalizada")
inputs = {}
for feature in X.columns:
    inputs[feature] = st.sidebar.number_input(feature, min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].mean()))

if st.sidebar.button("Predecir"):
    nueva_instancia = pd.DataFrame([inputs])
    prediccion = modelo.predict(nueva_instancia)
    resultado = class_names[int(prediccion[0])]
    st.sidebar.write(f"### Resultado: {resultado}")