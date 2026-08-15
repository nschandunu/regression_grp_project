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