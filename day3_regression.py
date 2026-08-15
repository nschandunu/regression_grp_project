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

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = m.predict(te[["size"]])
yte = te["price"]

print("MAE :", mean_absolute_error(yte, pred))
print("RMSE:", np.sqrt(mean_squared_error(yte, pred)))
print("R2  :", r2_score(yte, pred))

feats = ["size", "rooms", "age"]

m3 = LinearRegression().fit(tr[feats], tr["price"])

print("coefficients:", m3.coef_)
print("intercept   :", m3.intercept_)

pred3 = m3.predict(te[feats])

print("test R2 :", r2_score(te["price"], pred3))
print("test RMSE:", np.sqrt(mean_squared_error(te["price"], pred3)))

# Part E1 - Compare feature combinations

models = [
    ["size"],
    ["size", "rooms"],
    ["size", "rooms", "age"]
]

for feats in models:
    model = LinearRegression().fit(tr[feats], tr["price"])
    pred = model.predict(te[feats])
    r2 = r2_score(te["price"], pred)
    print(feats, "-> Test R2:", round(r2, 3))

# Part E2 - Add a useless feature

tr = tr.copy()
te = te.copy()

tr["noise"] = np.random.rand(len(tr))
te["noise"] = np.random.rand(len(te))

m4 = LinearRegression().fit(
    tr[["size", "rooms", "age", "noise"]],
    tr["price"]
)

pred4 = m4.predict(te[["size", "rooms", "age", "noise"]])

print(
    "with noise -> Test R2:",
    r2_score(te["price"], pred4)
)