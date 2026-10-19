"""Explore the insurance dataset: size, quality, statistics and patterns."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_FILE = Path(__file__).parent / "data" / "insurance.csv"
REPORTS = Path(__file__).parent / "reports"

df = pd.read_csv(DATA_FILE)

print("1. Size (rows, columns):", df.shape)
print("\n2. First 5 rows:")
print(df.head())

print("\n3. Missing values per column:")
print(df.isnull().sum())

print("\n4. Summary statistics for charges:")
print(df["charges"].describe().round(2))

print("\n5. Smokers vs non-smokers (count):")
print(df["smoker"].value_counts())

print("\n6. Average charges by smoker status:")
print(df.groupby("smoker")["charges"].mean().round(2))

print("\n7. Average charges by region:")
print(df.groupby("region")["charges"].mean().round(2))

print("\n8. Correlation of numeric columns with charges:")
print(df[["age", "bmi", "children", "charges"]].corr()["charges"].round(2))

threshold = df["charges"].quantile(0.75)
df["high_cost"] = (df["charges"] > threshold).astype(int)
print(f"\n9. High-cost threshold (75th percentile): {threshold:.2f}")
print("High-cost customers:", df["high_cost"].sum(), "of", len(df))

REPORTS.mkdir(exist_ok=True)
plt.hist(df["charges"], bins=40)
plt.title("Distribution of insurance charges")
plt.xlabel("Charges ($)")
plt.ylabel("Number of customers")
plt.savefig(REPORTS / "charges_histogram.png")
print("\nChart saved to reports/charges_histogram.png")