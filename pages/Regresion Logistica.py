import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Regresión Logística")
st.title("Análisis de Regresión Logística 📈")

# Cargar los datos automáticamente
df = pd.read_csv("csv/datos-logisticos.csv")
st.subheader("Datos Cargados:")
st.write(df)
# Definir las variables
X1 = 'Productos-Lote'
X2 = 'Tiempo-Entrega'
Y = 'Defectuoso'

# Separar los datos
X = df[[X1, X2]] 
y = df[Y]

# Entrenar el modelo
model = LogisticRegression(solver='liblinear')
model.fit(X, y)

# Crear una malla para evaluar el modelo
x_min, x_max = X[X1].min() - 0.5, X[X1].max() + 0.5
y_min, y_max = X[X2].min() - 0.5, X[X2].max() + 0.5
h = (x_max - x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico 1: Dispersión de los datos con la frontera de decisión
st.subheader("Gráficos de Dispersión:")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(X[y == 0][X1], X[y == 0][X2], color='blue', label='No Defectuoso')
ax1.scatter(X[y == 1][X1], X[y == 1][X2], color='red', label='Defectuoso')
ax1.contour(xx, yy, Z, cmap=plt.cm.Paired)
ax1.set_xlabel(X1)
ax1.set_ylabel(X2)
ax1.legend()
st.pyplot(fig1)

# Escalado y división de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# Entrenar nuevamente con los datos escalados
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilidades para la clase positiva
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Crear una malla para evaluar el modelo escalado
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
h = (x_max - x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Gráfico 2: Dispersión de datos escalados con frontera de decisión
st.subheader("Gráficos de Dispersión con Frontera de Decisión:")
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.5)
ax2.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], color='blue', label='No Defectuoso')
ax2.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], color='red', label='Defectuoso')
ax2.set_xlabel(X1)
ax2.set_ylabel(X2)
ax2.legend()
st.pyplot(fig2)

# Curva ROC y AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax4.set_xlabel("Tasa de Falsos Positivos")
ax4.set_ylabel("Tasa de Verdaderos Positivos")
ax4.set_title("Curva ROC")
ax4.legend(loc="lower right")
st.subheader("Curva ROC:")
st.pyplot(fig4)

st.subheader("Resultados del modelo")
st.write(f"**Exactitud (Accuracy):** {accuracy:.2f}")
st.write(f"**Precisión (Precision):** {precision:.2f}")
st.write(f"**Sensibilidad (Recall):** {recall:.2f}")
st.write(f"**F1-Score:** {f1:.2f}")
st.write("**Matriz de Confusión:**")
#st.write(conf_matrix)

# Visualización de la matriz de confusión
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["No Defectuoso", "Defectuoso"], yticklabels=["No Defectuoso", "Defectuoso"], ax=ax3)
ax3.set_ylabel("Etiqueta Real")
ax3.set_xlabel("Etiqueta Predicha")
ax3.set_title("Matriz de Confusión - Heatmap")
st.pyplot(fig3)

# Mostrar los coeficientes del modelo
st.subheader("Coeficientes del modelo")
coef_df = pd.DataFrame({"Variable": [X1, X2], "Coeficiente": model.coef_[0]})
st.write(coef_df)

# Entrada de datos en el sidebar
st.sidebar.header("Predicción personalizada")
Variable_x1 = st.sidebar.number_input("Productos en el lote", min_value=1, value=50)
Variable_x2 = st.sidebar.number_input("Tiempo de entrega (min)", min_value=1, value=80000)

if st.sidebar.button("Predecir"):
    new_example = pd.DataFrame([[Variable_x1, Variable_x2]], columns=[X1, X2])
    new_example_scaled = scaler.transform(new_example)
    prediction = model.predict(new_example_scaled)
    result_phrase = "DEFECTUOSO" if prediction[0] == 1 else "NO DEFECTUOSO"
    st.sidebar.write(f"### Resultado: {result_phrase}")
