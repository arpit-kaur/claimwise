"""Prepare the insurance data for machine learning and build a baseline."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).parent / "data" / "insurance.csv"


def load_data():
    """Load the data and add the high_cost target column."""
    df = pd.read_csv(DATA_FILE)
    threshold = df["charges"].quantile(0.75)
    df["high_cost"] = (df["charges"] > threshold).astype(int)
    return df


def get_train_test():
    """Return features and target, split into training and test sets."""
    df = load_data()
    X = df.drop(columns=["charges", "high_cost"])
    y = df["high_cost"]
    X = pd.get_dummies(X, columns=["sex", "smoker", "region"], drop_first=True)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()

    print("Columns the model will see:")
    print(list(X_train.columns))

    print("\nTraining rows:", len(X_train))
    print("Test rows:", len(X_test))
    print("Share of high-cost customers in training set:", round(y_train.mean(), 3))
    print("Share of high-cost customers in test set:", round(y_test.mean(), 3))

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    predictions = baseline.predict(X_test)

    print("\nBaseline accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Baseline recall:", round(recall_score(y_test, predictions), 3))