# Aplicación Web para Predecir Dureza del Hormigón

Voy a crear una aplicación web sencilla con Flask para el backend y un formulario HTML básico para el frontend que permita predecir la dureza del hormigón usando un modelo entrenado con el algoritmo XGBoost.

## Estructura del proyecto

```
hormigon_app/
│
├── app.py                  # Backend Flask
├── templates/
│   └── index.html          # Frontend básico
├── static/
├── final_model.pkl         # Modelo entrenado
└── scaler.pkl              # Scaler guardado
```

## Instrucciones para ejecutar la aplicación

1. Guarda los archivos en la estructura de carpetas indicada.
2. Asegúrate de tener instaladas las dependencias necesarias:
   ```
   pip install flask pandas numpy scikit-learn xgboost
   ```
3. Ejecuta la aplicación con:
   ```
   python app.py
   ```
4. Abre tu navegador en http://localhost:5000

## Funcionamiento

1. El usuario ingresa los parámetros del hormigón en el formulario web.
2. Al enviar el formulario, los datos se envían al backend Flask.
3. Flask procesa los datos:
   - Convierte los valores a float
   - Crea un DataFrame
   - Aplica el scaler a los datos
   - Usa el modelo para hacer la predicción
4. El resultado se muestra al usuario en la misma página.

La aplicación es muy básica pero completamente funcional. Puedes mejorarla añadiendo:
- Mensajes de error más descriptivos
- Gráficos o visualizaciones adicionales

