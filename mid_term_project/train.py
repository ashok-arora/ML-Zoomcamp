import pandas as pd
import seaborn as sns
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("./combined_dataset.csv")
df = df.drop(['track', 'artist', 'uri'], axis=1)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.target.values
del df_full_train["target"]

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient="records")
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient="records")
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(
    X_full_train, label=y_full_train, feature_names=dv.get_feature_names()
)

dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names())

xgb_params = {
    "eta": 0.08,
    "max_depth": 100,
    "min_child_weight": 2,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)

output_file = 'final_model_xgb.bin'

f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()
