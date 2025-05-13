from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf
print(tf.__version__)
from keras.models import load_model


app = Flask(__name__)

# Load model, scaler, encoder
model = load_model("iris_ann_model.h5")
scaler = joblib.load("iris_scaler.pkl")
encoder = joblib.load("iris_label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        features = np.array([features])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        class_index = np.argmax(pred)
        prediction = encoder.inverse_transform([class_index])[0]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
