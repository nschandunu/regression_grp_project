# Day 3 — Group Project: Build a Regression Model

*Teams of 3–4 · about 50 minutes to build*

---

## The goal

As a team, take a dataset **end to end**: explore it, build a regression model that predicts a number, test it honestly, and explain how good it is. You are not judged on getting the highest R² — you are judged on doing it **correctly** and being able to **explain** it.

You may use **Python (scikit-learn)** or **Octave (normal equation / polyfit)** — or both. Everything you need was covered in today's exercises.

---

## Choose a dataset

Pick one (or bring your own with at least ~100 rows and a numeric target):

- **`houses.csv`** — predict `price` from `size`, `rooms`, `age` (the workshop dataset; try to beat the exercise's R² ≈ 0.91 by engineering features).
- **scikit-learn `load_diabetes`** — predict disease progression from 10 health measurements. `from sklearn.datasets import load_diabetes`.
- **scikit-learn `fetch_california_housing`** — predict median house value from district features (needs internet the first time).
- **Your own data** — sports, weather, sales, study habits, anything with a number to predict.

> If you bring your own messy data, use your Day-1 cleaning skills first (nulls, duplicates, inconsistencies).

---

## What to do (the 6 steps)

1. **Pick & explore** — choose your dataset and target; look at shape, a few plots, and any missing values or outliers.
2. **Prepare & split** — choose your features and label; split into train and test (75/25, or first-most/last-rest).
3. **Train** — fit a regression model on the **training** set only.
4. **Evaluate** — report **RMSE** and **R²** on the **test** set; make a *predicted-vs-actual* plot.
5. **Interpret** — read your coefficients: which features matter, and do the signs make sense? Is the model good enough to use?
6. **Present** — what you built, how good it is, and what you'd improve.

---

## Deliverables

- Your **code** (a notebook or `.py`/`.m` files) that runs start to finish.
- A short **results summary**: your features, test RMSE and R², the predicted-vs-actual plot, and 3–4 sentences interpreting the model.

---

## Tips & common pitfalls

- **Never score on training data** — always report metrics on the held-back test set.
- **More features isn't automatically better** — add ones that make the *test* score rise; drop the rest.
- **Check the signs** of your coefficients against common sense — a wrong sign is a red flag.
- **Watch for outliers** — one crazy row can drag the whole line.
- **A modest model you understand and evaluate honestly beats a fancy one you can't explain.**

---

## Starter code

**Python**
```python
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("houses.csv")
feats = ["size", "rooms", "age"]        # <- choose your features
X, y = df[feats], df["price"]           # <- choose your target
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

m = LinearRegression().fit(Xtr, ytr)
pred = m.predict(Xte)
print("R2  :", r2_score(yte, pred))
print("RMSE:", np.sqrt(mean_squared_error(yte, pred)))
print("coefs:", dict(zip(feats, m.coef_)), "intercept:", m.intercept_)

plt.scatter(yte, pred); plt.plot(yte, yte)     # predicted vs actual
plt.xlabel("actual"); plt.ylabel("predicted"); plt.show()
```

**Octave**
```octave
data = csvread("houses.csv", 1, 0);
X = [ones(rows(data),1), data(:,1:3)];   % intercept + size, rooms, age
y = data(:,4);
ntr = round(0.75*rows(data));
b = (X(1:ntr,:)' * X(1:ntr,:)) \ (X(1:ntr,:)' * y(1:ntr));   % train
pred = X(ntr+1:end,:) * b;                                    % test
err  = pred - y(ntr+1:end);
RMSE = sqrt(mean(err.^2))
R2   = 1 - sum(err.^2)/sum((y(ntr+1:end)-mean(y(ntr+1:end))).^2)
```

