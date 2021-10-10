import pickle
import numpy as np


def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


with open("./model1.bin", "rb") as f_in:
    model = pickle.load(f_in)

with open("./dv.bin", "rb") as f_in:
    dv = pickle.load(f_in)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

prediction = predict_single(customer, dv, model)

print("\nprediction: %.3f" % prediction)

if prediction >= 0.5:
    print("verdict: Churn")
else:
    print("verdict: Not churn")
