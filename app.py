from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE.parent / "models" / "gdp.pkl"

app = Flask(__name__)
artifact = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if artifact is None:
        return render_template(
            "gdp_pred.html",
            prediction=None,
            error="Model not found. Run: python src/train_model.py"
        )

    values = {}
    for feature in artifact["feature_columns"]:
        raw = request.form.get(feature, "")
        if feature in artifact["encoders"]:
            try:
                values[feature] = artifact["encoders"][feature].transform([raw])[0]
            except ValueError:
                return render_template(
                    "gdp_pred.html",
                    prediction=None,
                    error=f"Unknown categorical value for {feature}: {raw}"
                )
        else:
            try:
                values[feature] = float(raw)
            except ValueError:
                return render_template(
                    "gdp_pred.html",
                    prediction=None,
                    error=f"Invalid numeric value for {feature}."
                )

    X = pd.DataFrame([values], columns=artifact["feature_columns"])
    prediction = float(artifact["model"].predict(X)[0])
    return render_template("gdp_pred.html", prediction=prediction, error=None)

if __name__ == "__main__":
    app.run(debug=True)
