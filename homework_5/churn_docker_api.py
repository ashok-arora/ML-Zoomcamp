from flask import Flask
from flask import request
from flask import jsonify
import pickle

with open("model2.bin", "rb") as f_in:
    model = pickle.load(f_in)

with open("dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

app = Flask("Churn Prediction")


@app.route("/predict", methods=["POST"])
def predict():
    customer_data = request.get_json()
    X = dv.transform([customer_data])
    y_pred = model.predict_proba(X)[:, 1]

    prediction = y_pred[0]
    if prediction >= 0.5:
        verdict = "Churn"
    else:
        verdict = "Not churn"

    result = {"churn_probability": prediction, "verdict": verdict}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
