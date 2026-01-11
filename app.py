from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["sleep"]),
        float(request.form["work"]),
        float(request.form["screen"]),
        float(request.form["activity"])
    ]

    prediction = model.predict([features])[0]

    stress_map = {
        0: "Low Stress",
        1: "Medium Stress",
        2: "High Stress"
    }

    result = stress_map[prediction]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
