# Day 3 — Regression Exercises (Python & Octave)
## Build, evaluate, and interpret regression models

*Dataset provided: **`houses.csv`** — 200 houses with columns `size` (100s of sq ft), `rooms`, `age` (years), and `price` ($1000s).*
*Do every exercise in **both** Python and Octave where shown. Solutions with expected numbers are at the end.*

**Shared split convention (so your numbers match the solutions):**
the file is already in random order, so we use the **first 150 rows to train** and the **last 50 rows to test**. Use this exact split in both tools.


---

## Part A — Warm-ups: load & look

### Python
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("houses.csv")
print(df.shape)          # (200, 4)
df.head()
df.describe()

plt.scatter(df["size"], df["price"])
plt.xlabel("size"); plt.ylabel("price"); plt.show()
```

### Octave
```octave
data = csvread("houses.csv", 1, 0);   % skip header row
size_  = data(:,1);
rooms  = data(:,2);
age    = data(:,3);
price  = data(:,4);

size(data)          % 200   4
mean(price)
scatter(size_, price); xlabel("size"); ylabel("price");
```

1. How many rows and columns?
2. What is the mean price? 
3. From the scatter of size vs price, does the relationship look positive, negative, or none?

---

## Part B — Simple linear regression (one feature: size)

Fit a line `price = intercept + slope × size` on the **training** rows.

### Python
```python
from sklearn.linear_model import LinearRegression

tr = df.iloc[:150]           # first 150 = train
te = df.iloc[150:]           # last 50   = test

X = tr[["size"]]; y = tr["price"]
m = LinearRegression().fit(X, y)
print("slope    :", m.coef_[0])
print("intercept:", m.intercept_)
print("predict size=18:", m.predict([[18]])[0])
```

### Octave
```octave
tr = 1:150; te = 151:200;
c = polyfit(size_(tr), price(tr), 1);   % degree-1 line
slope     = c(1)
intercept = c(2)
polyval(c, 18)                          % predict for size = 18
```

1. Report the slope and intercept. 
2. In one sentence each, what do they mean? 
3. Predict the price of a house with size 18. 
4. Confirm Python and Octave give the same numbers.

---

## Part C — Evaluate honestly (train/test + metrics)

Fit on train, score on the **test** rows with MAE, RMSE, R².

### Python
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pred = m.predict(te[["size"]])
yte  = te["price"]
print("MAE :", mean_absolute_error(yte, pred))
print("RMSE:", np.sqrt(mean_squared_error(yte, pred)))
print("R2  :", r2_score(yte, pred))
```

### Octave
```octave
pred = polyval(c, size_(te));
err  = pred - price(te);
MAE  = mean(abs(err))
RMSE = sqrt(mean(err.^2))
% R-squared:
ss_res = sum(err.^2);
ss_tot = sum((price(te) - mean(price(te))).^2);
R2 = 1 - ss_res/ss_tot
```

1. Report MAE, RMSE and R² for the size-only model. 
2. Is the RMSE small compared with typical prices? 3
3. What does an R² of about 0.24 tell you — is one feature enough?

---

## Part D — Multiple linear regression (size + rooms + age)

Now use all three features. In Octave we build the model with the **normal equation** — literally the matrix maths from Day 2.

### Python
```python
feats = ["size", "rooms", "age"]
m3 = LinearRegression().fit(tr[feats], tr["price"])
print("coefficients:", m3.coef_)      # one per feature
print("intercept   :", m3.intercept_)

pred3 = m3.predict(te[feats])
print("test R2 :", r2_score(te["price"], pred3))
print("test RMSE:", np.sqrt(mean_squared_error(te["price"], pred3)))
```

### Octave — build X as a matrix, solve for the weights
```octave
% design matrix: a column of ones (intercept) + the 3 features
Xtr = [ones(150,1), size_(tr), rooms(tr), age(tr)];
b   = (Xtr' * Xtr) \ (Xtr' * price(tr));   % normal equation
% b = [intercept; w_size; w_rooms; w_age]

Xte  = [ones(50,1), size_(te), rooms(te), age(te)];
pred = Xte * b;                            % one dot product per row
err  = pred - price(te);
RMSE = sqrt(mean(err.^2))
R2   = 1 - sum(err.^2)/sum((price(te)-mean(price(te))).^2)
b
```

1. Report the three coefficients and the intercept. 
2. Interpret each coefficient in plain words (which pushes price up, which down?). 
3. Report the test R² and RMSE. 
4. Compare with the size-only model from Part C — how much did adding rooms and age help? 
5. Do Python and Octave agree?

---

## Part E — Challenge: features vs complexity

1. Fit three models and record the test R²: (a) size only, (b) size + rooms, (c) size + rooms + age. Make a small table. What happens to R² as you add relevant features?

2. Add a **useless** feature — a column of random numbers — and refit the full model.
```python
tr = tr.copy(); te = te.copy()
tr["noise"] = np.random.rand(len(tr))
te["noise"] = np.random.rand(len(te))
m4 = LinearRegression().fit(tr[["size","rooms","age","noise"]], tr["price"])
print("test R2:", r2_score(te["price"], m4.predict(te[["size","rooms","age","noise"]])))
```
Does the test R² improve? What does this tell you about throwing in features that aren't relevant? (Link back to over-fitting and the "complexity dial".)

3. In one paragraph: is your final model good enough to actually use? What would you improve, and what are its limits?

---

## Part F — Deciding to drop or create a feature

How do you know which features to keep, drop, or invent? **Test by ablation**: remove one feature, refit, and see how much the test score falls. If it barely moves, that feature wasn't pulling its weight.

### F1 — Which features matter? (drop-one test, Python)
```python
from sklearn.metrics import r2_score
def test_r2(feats):
    mm = LinearRegression().fit(tr[feats], tr["price"])
    return round(r2_score(te["price"], mm.predict(te[feats])), 3)

print("all three     :", test_r2(["size","rooms","age"]))
print("drop size     :", test_r2(["rooms","age"]))
print("drop rooms    :", test_r2(["size","age"]))
print("drop age      :", test_r2(["size","rooms"]))
```
**Question:** which feature, when removed, hurts the score the most? That's your most important feature. Should you drop any of them?

### F2 — Create a feature (feature engineering)
Invent a new column and see if it earns a place.
```python
tr = tr.copy(); te = te.copy()
tr["size_per_room"] = tr["size"] / tr["rooms"]
te["size_per_room"] = te["size"] / te["rooms"]
print("with size_per_room:", test_r2(["size","rooms","age","size_per_room"]))

tr["size_x_rooms"] = tr["size"] * tr["rooms"]
te["size_x_rooms"] = te["size"] * te["rooms"]
print("with size*rooms   :", test_r2(["size","rooms","age","size_x_rooms"]))
```
**Question:** did either new feature improve the test R²? Given how this data was built (price is a straight-line, additive mix of the features), why might engineered features add little here — and when *would* they help?

### Octave — the same drop-one test
```octave
function r2 = fit_r2(Xtr, ytr, Xte, yte)
  b = (Xtr'*Xtr)\(Xtr'*ytr);
  pred = Xte*b;
  r2 = 1 - sum((yte-pred).^2)/sum((yte-mean(yte)).^2);
end
tr = 1:150; te = 151:200;
o = ones(150,1); oe = ones(50,1);
full  = fit_r2([o size_(tr) rooms(tr) age(tr)], price(tr), [oe size_(te) rooms(te) age(te)], price(te))
noAge = fit_r2([o size_(tr) rooms(tr)],          price(tr), [oe size_(te) rooms(te)],          price(te))
```

**F3.** Write a 2–3 sentence rule of thumb your team would follow for deciding to keep, drop, or create a feature.

---

## Part G — Fine-tuning (hyperparameters)

The model *learns* its weights (parameters). **You** choose the *hyperparameters* — settings like how much to regularise. Here you tune the regularisation strength `alpha` and estimate performance honestly with cross-validation.

### G1 — Cross-validation for a robust score (Python)
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(LinearRegression(), tr[["size","rooms","age"]],
                         tr["price"], cv=5, scoring="r2")
print("fold scores:", np.round(scores, 2))
print("mean CV R2 :", round(scores.mean(), 3))
```
**Question:** how much do the 5 fold scores vary? Why is the average a more trustworthy estimate than a single split?

### G2 — Tune the regularisation strength (Python)
```python
from sklearn.linear_model import Ridge, RidgeCV

# let cross-validation pick alpha for us
rc = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5).fit(tr[["size","rooms","age"]], tr["price"])
print("chosen alpha:", rc.alpha_)
print("test R2     :", round(r2_score(te["price"], rc.predict(te[["size","rooms","age"]])), 3))

# see what a big alpha does to the weights
for a in [0, 1, 10, 100]:
    m = Ridge(alpha=a).fit(tr[["size","rooms","age"]], tr["price"])
    print(f"alpha={a:>4}  coefs={np.round(m.coef_,2)}  "
          f"testR2={round(r2_score(te['price'], m.predict(te[['size','rooms','age']])),3)}")
```
**Question:** what alpha did cross-validation choose? As alpha grows very large, what happens to the coefficients and the test R²? Did regularisation improve this model — and why or why not?

### Octave — ridge is one small change to the normal equation
```octave
tr = 1:150; te = 151:200;
X = [ones(150,1), size_(tr), rooms(tr), age(tr)];
y = price(tr);
lambda = 1;
I = eye(4); I(1,1) = 0;                 % don't penalise the intercept
b = (X'*X + lambda*I) \ (X'*y);         % ridge normal equation
Xte  = [ones(50,1), size_(te), rooms(te), age(te)];
pred = Xte * b;
R2 = 1 - sum((price(te)-pred).^2)/sum((price(te)-mean(price(te))).^2)
b
```
Notice ridge just adds `lambda*I` to `X'X` — bigger `lambda` shrinks the weights.

**G3.** In your own words: what is the difference between a *parameter* and a *hyperparameter*? Give one example of each from this exercise.

---

## Part H — Polynomial regression (fitting curves)

Linear regression can fit **curves** too — you just feed it `x²` (and higher powers) as extra features. It's still "linear" because it's linear in the *weights*. Here you fit a clearly curved dataset and watch the **degree** act as a complexity dial.

### Make curvy data (Python)
```python
import numpy as np
from numpy.polynomial import polynomial
from sklearn.metrics import r2_score
rng = np.random.default_rng(5)
x = rng.uniform(-3, 3, 120)
y = 2 + 0.6*x + 0.9*x**2 + rng.normal(0, 1.6, 120)   # a U-shaped (quadratic) trend
order = rng.permutation(120); tr, te = order[:90], order[90:]
```

### Fit different degrees and compare (Python)
```python
for d in [1, 2, 3, 9, 15]:
    c = np.polyfit(x[tr], y[tr], d)              # degree-d polynomial
    r2 = r2_score(y[te], np.polyval(c, x[te]))   # test score
    print(f"degree {d:>2}  test R2 = {round(r2,3)}")
```

### Octave — same idea, `polyfit` takes a degree
```octave
% (load or generate x, y first)
c1 = polyfit(x, y, 1);   % straight line
c2 = polyfit(x, y, 2);   % quadratic curve
% plot both over the data:
xg = linspace(min(x), max(x), 100);
plot(x, y, 'o'); hold on;
plot(xg, polyval(c1, xg), 'r--');   % underfits
plot(xg, polyval(c2, xg), 'g-');    % fits the curve
```

**H1.** Which degree fits best on the test set? What happens at degree 1 (too low) and degree 15 (too high)? **H2.** Plot the degree-1 and degree-2 fits over the data — describe what you see. **H3.** Why is polynomial regression still called *linear* regression? **H4.** How does the *degree* relate to over- and under-fitting (link to the complexity dial)?

