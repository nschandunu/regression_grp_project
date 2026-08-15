import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ---------- 1. Pick & explore ----------
df = pd.read_csv("House_price.csv")
print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary stats:\n", df.describe())

# Quick correlation with target — helps sanity-check signs later
print("\nCorrelation with Price:\n", df.corr(numeric_only=True)["Price"].sort_values(ascending=False))

# ---------- 2. Prepare & split ----------
feats = ["Avg. Area Income", "House Age", "Number of Rooms",
          "Number of Bedrooms", "Area Population"]
X, y = df[feats], df["Price"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=1)

# ---------- 3. Train ----------
model = LinearRegression().fit(Xtr, ytr)

# ---------- 4. Evaluate (on TEST set only) ----------
pred = model.predict(Xte)
rmse = np.sqrt(mean_squared_error(yte, pred))
r2 = r2_score(yte, pred)

print("\n--- Test set performance ---")
print(f"RMSE: {rmse:,.2f}")
print(f"R2  : {r2:.4f}")

print("\n--- Coefficients ---")
for f, c in zip(feats, model.coef_):
    print(f"{f:>22}: {c:>12,.2f}")
print(f"{'Intercept':>22}: {model.intercept_:>12,.2f}")

# Predicted-vs-actual plot
plt.figure(figsize=(6, 6))
plt.scatter(yte, pred, alpha=0.5, edgecolor="k", linewidth=0.3)
lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
plt.plot(lims, lims, "r--", label="perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Predicted vs Actual (Test set)\nR2={r2:.3f}, RMSE={rmse:,.0f}")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved plot to predicted_vs_actual.png")
