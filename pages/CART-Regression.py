import streamlit as st
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Árbol de Regresión")
st.title("Análisis de Árbol de Regresión 🌳")

df = pd.read_csv("csv/gym_members_exercise_tracking.csv")
st.subheader("Datos Cargados:")
st.write(df)

categorical_features = ["Gender", "Workout_Type"]
numerical_features = [
    "Age", "Weight (kg)", "Height (m)", "Max_BPM", "Avg_BPM",
    "Resting_BPM", "Session_Duration (hours)", "Fat_Percentage",
    "Water_Intake (liters)", "Workout_Frequency (days/week)", "Experience_Level", "BMI"
]
target = "Calories_Burned"

X = df[categorical_features + numerical_features]
y = df[target]

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), categorical_features)],
    remainder="passthrough"
)
X = ct.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Entrenar el modelo
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

st.subheader("Evaluación del Modelo")
st.write(f"**Error Cuadrático Medio (MSE):** {mse:.2f}")
st.write(f"**Raíz del Error Cuadrático Medio (RMSE):** {rmse:.2f}")

# Gráfico: Valores reales vs Predicciones
st.subheader("Valores Reales vs Predicciones")
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.scatter(y_test, y_pred, alpha=0.7, color='blue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel("Valores Reales")
ax1.set_ylabel("Predicciones")
ax1.set_title("Predicciones vs Valores Reales")
st.pyplot(fig1)

# Gráfico del Árbol
st.subheader("Visualización del Árbol de Decisión")
from sklearn.tree import plot_tree
fig2, ax2 = plt.subplots(figsize=(30, 15))
plot_tree(model, filled=True, feature_names=ct.get_feature_names_out(), ax=ax2)
buffer = io.BytesIO()
fig2.savefig(buffer, format='png')
buffer.seek(0)  # Mover el puntero al inicio del archivo

# Descargar el gráfico
st.download_button(
    label="Descargar Gráfico del Árbol",
    data=buffer,
    file_name="arbol_decision.png",
    mime="image/png"
)

# Cerrar la figura para liberar memoria
plt.close(fig2)

# Histograma de los errores
st.subheader("Distribución de Errores")
errors = y_test - y_pred
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.histplot(errors, bins=20, kde=True, ax=ax3, color='purple')
ax3.set_title("Distribución de Errores")
ax3.set_xlabel("Error (Valores Reales - Predicciones)")
ax3.set_ylabel("Frecuencia")
st.pyplot(fig3)

# Entrada personalizada para predicción
st.sidebar.header("Predicción Personalizada")
input_values = []

# Agregar entrada para cada característica
for feature in numerical_features:
    input_values.append(
        st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
    )

# Seleccionar categorías para las variables categóricas
for cat in categorical_features:
    unique_values = df[cat].unique()
    selected = st.sidebar.selectbox(f"Selecciona {cat}", unique_values)
    input_values.append(selected)  # Pasar el texto original

if st.sidebar.button("Predecir"):
    try:
        input_df = pd.DataFrame([input_values], columns=numerical_features + categorical_features)
        for col in categorical_features:
            input_df[col] = input_df[col].astype(str)
        input_transformed = ct.transform(input_df)
        input_scaled = scaler.transform(input_transformed)
        prediction = model.predict(input_scaled)
        st.subheader("Resultado de la Predicción")
        st.write(f"### Calorías quemadas estimadas: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")