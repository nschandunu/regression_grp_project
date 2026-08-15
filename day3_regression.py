import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("houses.csv")

print(df.shape)
print(df.head())
print(df.describe())

plt.scatter(df["size"], df["price"])
plt.xlabel("size")
plt.ylabel("price")
plt.show()

from sklearn.linear_model import LinearRegression

tr = df.iloc[:150]   # first 150 = training data
te = df.iloc[150:]   # last 50 = testing data

X = tr[["size"]]
y = tr["price"]

m = LinearRegression().fit(X, y)

print("slope    :", m.coef_[0])
print("intercept:", m.intercept_)
print("predict size=18:", m.predict([[18]])[0])