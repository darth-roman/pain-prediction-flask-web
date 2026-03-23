from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://darth-roman.github.io"], methods=["GET", "POST", "OPTIONS"], supports_credentials=True,  allow_headers=["Content-Type", "Authorization"])

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

PAIN_LEVELS = {
    1: "Minimal",
    2: "Mild",
    3: "Mild-Moderate",
    4: "Moderate",
    5: "Moderate-Severe",
    6: "Severe",
    7: "Intense",
    8: "Very Intense",
    9: "Worst Possible",
    10: "Maximum Pain",
}


# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        float(data["eda"]),
        float(data["bvp"]),
        float(data["hr"]),
        float(data["temp"]),
    ]])

    features_scaled = scaler.transform(features)

    print(features_scaled)
    prediction = int(model.predict(features_scaled)[0])

    label = PAIN_LEVELS.get(prediction, f"Pain scale {prediction}")

    return jsonify({
        "pain_scale": prediction,
        "pain_level": label
    })


if __name__ == "__main__":
    app.run(debug=False)