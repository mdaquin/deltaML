import pandas as pd
import numpy as np
from deltaml import DiffLearning
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import time

# https://repositorium.sdum.uminho.pt/bitstream/1822/8024/1/student.pdf
# https://www.kaggle.com/datasets/impapan/student-performance-data-set
# df = pd.read_csv("../mdaquin.github.io/d/student-mat.csv", sep=";")
df = pd.read_csv("../mdaquin.github.io/d/student-por.csv", sep=";")

topredict = "G3"
#toremove = ["G1", "G2"]
toremove = []

numcols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]

df.drop(toremove, axis=1, inplace=True)

schools = pd.get_dummies(df.school, prefix='school')
df.drop(["school"], axis=1, inplace=True)
sexes = pd.get_dummies(df.sex, prefix='sex')
df.drop(["sex"], axis=1, inplace=True)
add = pd.get_dummies(df.address, prefix='address')
df.drop(["address"], axis=1, inplace=True)
fs = pd.get_dummies(df.famsize, prefix='famsize')
df.drop(["famsize"], axis=1, inplace=True)
ps = pd.get_dummies(df.Pstatus, prefix='Pstatus')
df.drop(["Pstatus"], axis=1, inplace=True)
mjob = pd.get_dummies(df.Mjob, prefix='Mjob')
df.drop(["Mjob"], axis=1, inplace=True)
fjob = pd.get_dummies(df.Fjob, prefix='Fjob')
df.drop(["Fjob"], axis=1, inplace=True)
reason = pd.get_dummies(df.reason, prefix='reason')
df.drop(["reason"], axis=1, inplace=True)
guardian = pd.get_dummies(df.guardian, prefix='guardian')
df.drop(["guardian"], axis=1, inplace=True)
ssup = pd.get_dummies(df.schoolsup, prefix='schoolsup')
df.drop(["schoolsup"], axis=1, inplace=True)
fsup = pd.get_dummies(df.famsup, prefix='famsup')
df.drop(["famsup"], axis=1, inplace=True)
paid = pd.get_dummies(df.paid, prefix='paid')
df.drop(["paid"], axis=1, inplace=True)
activities = pd.get_dummies(df.activities, prefix='activities')
df.drop(["activities"], axis=1, inplace=True)
nursery = pd.get_dummies(df.nursery, prefix='nursery')
df.drop(["nursery"], axis=1, inplace=True)
higher = pd.get_dummies(df.higher, prefix='higher')
df.drop(["higher"], axis=1, inplace=True)
internet = pd.get_dummies(df.internet, prefix='internet')
df.drop(["internet"], axis=1, inplace=True)
romantic = pd.get_dummies(df.romantic, prefix='romantic')
df.drop(["romantic"], axis=1, inplace=True)
df = pd.concat([df, schools, sexes, add, fs, ps, mjob, fjob, reason, guardian, ssup, fsup, paid, activities, nursery, higher, internet, romantic], axis=1)

df_s = df.copy()

for c in numcols:
   df_s[c] = (df[c] - df[c].mean()) / df[c].std()
df_s[topredict] =  (df[topredict] - df[topredict].mean()) / df[topredict].std()

X = df_s.copy().drop([topredict], axis=1)
y = df_s[topredict]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=P["test_size"], random_state=1)

n = P["nbnn_train"]
dl = DiffLearning(X_train, y_train, neighbors=n, context=True)
X_train_d, y_train_d = dl.diffDataset()
