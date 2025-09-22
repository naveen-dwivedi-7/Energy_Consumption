# app.py
from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
with open("energy_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # --- Manual input ---
    if request.form.get('manual') == 'yes':
        # You can fill some default/fixed values here
        data = {
            'Temperature': float(request.form.get('Temperature', 25.0)),  # default 25
            'Humidity': float(request.form.get('Humidity', 50.0)),        # default 50
            'SquareFootage': float(request.form.get('SquareFootage', 1500)),
            'Occupancy': int(request.form.get('Occupancy', 5)),
            'HVACUsage': int(request.form.get('HVACUsage', 1)),           # 1=On, 0=Off
            'LightingUsage': int(request.form.get('LightingUsage', 1)),   # 1=On, 0=Off
            'RenewableEnergy': float(request.form.get('RenewableEnergy', 2.0)),
            'Hour': int(request.form.get('Hour', 12)),
            'Month': int(request.form.get('Month', 1)),
            'Day': int(request.form.get('Day', 1))
        }
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return render_template("result.html", prediction=prediction, method="Manual Input")

    # --- CSV Upload ---
    file = request.files.get('file')
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        df = pd.read_csv(filepath)

        # Extract datetime features
        if 'Timestamp' in df.columns:
            df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
            df['Month'] = pd.to_datetime(df['Timestamp']).dt.month
            df['Day'] = pd.to_datetime(df['Timestamp']).dt.day

        # Drop unnecessary or target columns
        df = df.drop(columns=['Timestamp', 'DayOfWeek', 'Holiday', 'EnergyConsumption'], errors='ignore')

        # Encode categorical columns
        for col in ['HVACUsage','LightingUsage']:
            if col in df.columns:
                df[col] = df[col].map({'On':1, 'Off':0})

        predictions = model.predict(df)
        df['Predicted_Energy'] = predictions
        return render_template("result.html", table=df.to_html(), method="File Upload")

    return "No input provided"


if __name__ == "__main__":
    app.run(debug=True)
