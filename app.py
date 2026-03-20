from flask import Flask, request, render_template
import joblib
import numpy as np
import os 

# Initialize app
app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(final_features)[0]

        return render_template("index.html", prediction_text=f"Predicted Wine Quality: {round(prediction,2)}")

    except Exception as e:
        return str(e)

# Run app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)