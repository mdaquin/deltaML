import pandas as pd
import numpy as np
from deltaml import DiffLearning
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import time

df = pd.read_csv("https://mdaquin.github.io/d/AirfoilSelfNoise.csv")

df_s = df.copy()
for k in df:
  df_s[k] = (df[k] - df[k].mean()) / df[k].std() 

from sklearn.model_selection import train_test_split
X = df_s.copy().drop(["SSPL"], axis=1)
y = df_s.SSPL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=P["test_size"], random_state=1)

n = P["nntrain"]
dl = DiffLearning(X_train, y_train, neighbors=n)
X_train_d, y_train_d = dl.diffDataset()
print(X_train_d.shape)
print(y_train_d.shape)

