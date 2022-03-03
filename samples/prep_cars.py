import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error
import matplotlib.pyplot as plt
import time

# uncomment the line for the data of the brand of car to use
# df = pd.read_csv("https://mdaquin.github.io/d/cars/toyota.csv") # ~96%
# df = pd.read_csv("https://mdaquin.github.io/d/cars/audi.csv") # ~94.5
# df = pd.read_csv("https://mdaquin.github.io/d/cars/bmw.csv") # ~94%
# df = pd.read_csv("https://mdaquin.github.io/d/cars/ford.csv") # ~93%
# df = pd.read_csv("https://mdaquin.github.io/d/cars/merc.csv") # ~94%
df = pd.read_csv("https://mdaquin.github.io/d/cars/vauxhall.csv") # ~86.5
# df = pd.read_csv("https://mdaquin.github.io/d/cars/vw.csv") # ~95.5

# one hot encoding of categorical values
models = pd.get_dummies(df.model, prefix='model')
ft = pd.get_dummies(df.fuelType)
tr = pd.get_dummies(df.transmission)
# put it all together and remove the orginal columns
df_m=pd.concat([models, ft, tr, df], axis=1).drop(["model", "fuelType", "transmission"], axis=1)

# standardization by the mean of numerical values
# it is slower and does not quick get the same results
# without this
df_ms = df_m.copy()
df_ms["price"] = (df_m.price - df_m.price.mean()) / df_m.price.std()
df_ms["year"] = (df_m.year - df_m.year.mean()) / df_m.year.std()
df_ms["mileage"] = (df_m.mileage - df_m.mileage.mean()) / df_m.mileage.std()
df_ms["tax"] = (df_m.tax - df_m.tax.mean()) / df_m.tax.std()
df_ms["mpg"] = (df_m.mpg - df_m.mpg.mean()) / df_m.mpg.std()
df_ms["engineSize"] = (df_m.engineSize - df_m.engineSize.mean()) / df_m.engineSize.std()

X = df_ms.copy().drop(["price"], axis=1)
y = df_ms.price
# parameter: test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=P["test_size"], random_state=1)

