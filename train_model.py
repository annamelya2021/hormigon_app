import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Cargar el conjunto de datos de precios de viviendas de California
print("Cargando datos...")
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Dividir los datos en conjuntos de entrenamiento y prueba
print("Dividiendo datos...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
print("Escalando datos...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo
print("Entrenando modelo...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluar el modelo
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)
print(f"\nPrecisión del modelo:")
print(f"- Conjunto de entrenamiento: {train_score:.4f}")
print(f"- Conjunto de prueba: {test_score:.4f}")

# Guardar el modelo y el escalador
print("\nGuardando modelo y escalador...")
with open('static/housing_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('static/housing_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n¡Modelo entrenado y guardado exitosamente!")
print("Archivos generados:")
print("- static/housing_model.pkl")
print("- static/housing_scaler.pkl")
