from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el scaler al iniciar la aplicación
model = pickle.load(open('static/final_model.pkl', 'rb'))
scaler = pickle.load(open('static/scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Obtener datos del formulario
            cement = float(request.form['cement'])
            slag = float(request.form['slag'])
            ash = float(request.form['ash'])
            water = float(request.form['water'])
            superplastic = float(request.form['superplastic'])
            coarseagg = float(request.form['coarseagg'])
            fineagg = float(request.form['fineagg'])
            age = float(request.form['age'])
            
            # Crear DataFrame con los datos de entrada
            input_data = pd.DataFrame({
                'cement': [cement],
                'slag': [slag],
                'ash': [ash],
                'water': [water],
                'superplastic': [superplastic],
                'coarseagg': [coarseagg],
                'fineagg': [fineagg],
                'age': [age]
            })
            
            # Escalar los datos
            scaled_data = scaler.transform(input_data)
            
            # Hacer la predicción
            prediction = model.predict(scaled_data)
            
            # Redondear a 2 decimales
            prediction_rounded = round(prediction[0], 2)
            
            return render_template('index.html', 
                                prediction_text=f'La dureza estimada del hormigón es: {prediction_rounded} MPa',
                                show_result=True)
            
        except Exception as e:
            return render_template('index.html', 
                                prediction_text=f'Error: {str(e)}',
                                show_result=True)
    
    # Si es GET, mostrar el formulario vacío
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)