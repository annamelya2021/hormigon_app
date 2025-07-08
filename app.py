from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Cargar el conjunto de datos para obtener las características
california = fetch_california_housing()
feature_names = california.feature_names

# Crear y entrenar un modelo simple (en producción, deberías cargar un modelo pre-entrenado)
# Esto es solo para demostración
scaler = StandardScaler()
X = california.data
y = california.target
X_scaled = scaler.fit_transform(X)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            med_inc = float(request.form['med_inc'])
            house_age = float(request.form['house_age'])
            avg_rooms = float(request.form['avg_rooms'])
            avg_bedrms = float(request.form['avg_bedrms'])
            population = float(request.form['population'])
            avg_occup = float(request.form['avg_occup'])
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            
            # Crear DataFrame con los datos de entrada
            input_data = pd.DataFrame({
                'MedInc': [med_inc],
                'HouseAge': [house_age],
                'AveRooms': [avg_rooms],
                'AveBedrms': [avg_bedrms],
                'Population': [population],
                'AveOccup': [avg_occup],
                'Latitude': [latitude],
                'Longitude': [longitude]
            })
            
            # Escalar los datos
            scaled_data = scaler.transform(input_data)
            
            # Hacer la predicción
            prediction = model.predict(scaled_data)
            
            # Formatear el resultado
            prediction_formatted = f'${prediction[0]*100000:,.2f}'
            
            return render_template('index.html', 
                                prediction_text=f'El precio estimado de la vivienda es: {prediction_formatted}',
                                show_result=True)
            
        except Exception as e:
            return render_template('index.html', 
                                prediction_text=f'Error: {str(e)}',
                                show_result=True)
    
    # Si es GET, mostrar el formulario vacío
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)