from flask import Flask
from flask import request
from flask import jsonify
import pickle
import xgboost as xgb


with open("final_model_xgb.bin", "rb") as f_in:
    (dv, model) = pickle.load(f_in)


app = Flask("Spotify Hit Prediction")


@app.route("/", methods=["GET"])
def greet():
    return "Welcome to Spotify Hit Prediction API. POST song data to /predict endpoint to get prediction stats."

@app.route("/predict", methods=["POST"])
def predict():
    song_data = request.get_json()

    # song_data = json.loads(request.get_json())
    # print(type(song_data))
    X = dv.transform([song_data])
    dX = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dX)
    
    print(y_pred)
    prediction = y_pred[0]
    print(prediction)
    if prediction >= 0.5:
        verdict = "Hit"
    else:
        verdict = "Flop"

    result = {"hit_probability": float(prediction), "verdict": verdict}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
