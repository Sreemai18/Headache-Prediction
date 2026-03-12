from flask import Flask, render_template, request
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("model/headache_model.pkl")

# Feature scaler (must match training)
scaler = StandardScaler()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    sleep_duration = float(request.form["sleep"])
    quality_sleep = float(request.form["quality"])
    physical_activity = float(request.form["activity"])
    stress = float(request.form["stress"])
    heart_rate = float(request.form["heart"])
    steps = float(request.form["steps"])
    systolic = float(request.form["systolic"])
    diastolic = float(request.form["diastolic"])

    sleep_stress_index = stress / sleep_duration

    features = np.array([[age,
                          sleep_duration,
                          quality_sleep,
                          physical_activity,
                          stress,
                          heart_rate,
                          steps,
                          systolic,
                          diastolic,
                          sleep_stress_index]])

    prediction = model.predict(features)

    if prediction[0] == 0:
        result = "Moderate Headache Risk"
    else:
        result = "High Headache Risk"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
