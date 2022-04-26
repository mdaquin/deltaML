import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
from deltaml import DiffLearning

# https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction?select=economy.csv
# df = pd.read_csv("https://mdaquin.github.io/d/flights/economy.csv")
df = pd.read_csv("https://mdaquin.github.io/d/flights/business.csv")

df = df.sample(15000, random_state = 1)

# date to int
df["date"] = pd.to_datetime(df.date).dt.strftime("%Y%m%d").astype(int)

airlines = pd.get_dummies(df.airline, prefix='airline')
df.drop(["airline"], axis=1, inplace=True)

# ch_code redundant with airline
df.drop(["ch_code"], axis=1, inplace=True)
# numcode irrelevant
df.drop(["num_code"], axis=1, inplace=True)

froms = pd.get_dummies(df["from"], prefix='from')
df.drop(["from"], axis=1, inplace=True)

deptime = pd.to_datetime(df.dep_time).dt
df["dep_time"] = deptime.strftime("%H").astype(int)*60+deptime.strftime("%M").astype(int)

def convertDateTime(x):
  h = 0
  m = 0
  try:
     h = int(x[:x.index("h")])
     m = int(x[x.index(" ")+1:x.index("m")])
  except: 
    pass
  return h*60+m

df["time_taken"] = df.time_taken.apply(convertDateTime)

df["price"] = df.price.apply(lambda x : x.replace(",", "")).astype(int)

def convertStops(x):
  if "non-stop" in x: return 0
  elif "1-stop" in x: return 1
  return 2

df["stop"] = df.stop.apply(convertStops)

arrtime = pd.to_datetime(df.arr_time).dt
df["arr_time"] = arrtime.strftime("%H").astype(int)*60+arrtime.strftime("%M").astype(int)

tos = pd.get_dummies(df["to"], prefix='to')
df.drop(["to"], axis=1, inplace=True)

df = pd.concat([df, airlines, froms, tos], axis=1)

df_s = df.copy()
for k in df:
  df_s[k] = (df[k] - df[k].mean()) / df[k].std()

X = df_s.copy().drop(["price"], axis=1)
y = df_s.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=P["test_size"], random_state=1)

n = P["nbnn_train"]
dl = DiffLearning(X_train, y_train, neighbors=n, context=True)
X_train_d, y_train_d = dl.diffDataset()
