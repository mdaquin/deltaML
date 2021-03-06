import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import time

df = pd.read_csv("data/AirfoilSelfNoise.csv")

df_s = df.copy()
for k in df:
  df_s[k] = (df[k] - df[k].mean()) / df[k].std() 

X = df_s.copy().drop(["SSPL"], axis=1)
y = df_s.SSPL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=P["test_size"], random_state=1)


