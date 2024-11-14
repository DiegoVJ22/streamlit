import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Entrenar el modelo de Árbol de Regresión
tree_model = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Métricas para el Árbol de Regresión
mse_tree = mean_squared_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mse_tree)
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)

# Entrenar el modelo de Regresión Lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Métricas para la Regresión Lineal
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Crear un DataFrame con las métricas de comparación
comparison_data = {
    "Métrica": [
        "Error Cuadrático Medio (MSE)",
        "Raíz del Error Cuadrático Medio (RMSE)",
        "Error Absoluto Medio (MAE)",
        "Coeficiente de Determinación (R²)"
    ],
    "Árbol de Decisión": [
        f"{mse_tree:.2f}",
        f"{rmse_tree:.2f}",
        f"{mae_tree:.2f}",
        f"{r2_tree:.2f}"
    ],
    "Regresión Lineal": [
        f"{mse_linear:.2f}",
        f"{rmse_linear:.2f}",
        f"{mae_linear:.2f}",
        f"{r2_linear:.2f}"
    ]
}

comparison_df = pd.DataFrame(comparison_data)
# Mostrar la tabla en Streamlit
st.subheader("Comparación de Resultados entre Modelos")
st.dataframe(comparison_df)
st.write('La regresión lineal múltiple es más eficiente en este caso porque los datos tienen una relación predominantemente lineal con la variable objetivo, lo que permite que este modelo capture mejor las dependencias.')

# Gráfico del Árbol
st.subheader("Visualización del Árbol de Decisión")
from sklearn.tree import plot_tree
fig2, ax2 = plt.subplots(figsize=(20, 10))
plot_tree(tree_model, filled=True, feature_names=ct.get_feature_names_out(), ax=ax2)
st.pyplot(fig2)

from sklearn.tree import export_text
tree_rules = export_text(tree_model, feature_names=ct.get_feature_names_out())
with st.expander("Transcripción del árbol"):
    st.text(tree_rules)

# Comparación visual de valores reales y predicciones
st.subheader("Comparación de Valores Reales y Predicciones")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, y_pred_tree, alpha=0.7, color='blue', label='Árbol de Decisión')
ax.scatter(y_test, y_pred_linear, alpha=0.7, color='green', label='Regresión Lineal')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel("Valores Reales")
ax.set_ylabel("Predicciones")
ax.legend()
ax.set_title("Comparación de Modelos")
st.pyplot(fig)

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
        tree_prediction = tree_model.predict(input_scaled)
        linear_prediction = linear_model.predict(input_scaled)
        st.subheader("Resultado de la Predicción: Regression Tree")
        st.write(f"Calorías quemadas estimadas: **{tree_prediction[0]:.2f}**")
        st.subheader("Resultado de la Predicción: Linear Regression")
        st.write(f"Calorías quemadas estimadas: **{linear_prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")