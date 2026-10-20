"""Train two models to predict high-cost customers, compare them, and save the best."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from prepare import get_train_test

MODELS = Path(__file__).parent / "models"


def evaluate(name, model, X_train, X_test, y_train, y_test):
    """Print scores for one model and return its F1 score on the test set."""
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print("Train accuracy:", round(accuracy_score(y_train, train_pred), 3))
    print("Test accuracy: ", round(accuracy_score(y_test, test_pred), 3))
    print("Precision:     ", round(precision_score(y_test, test_pred), 3))
    print("Recall:        ", round(recall_score(y_test, test_pred), 3))
    print("F1 score:      ", round(f1_score(y_test, test_pred), 3))
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, test_pred))
    return f1_score(y_test, test_pred)


X_train, X_test, y_train, y_test = get_train_test()

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_f1 = evaluate("Logistic Regression", log_model, X_train, X_test, y_train, y_test)

forest = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
forest.fit(X_train, y_train)
forest_f1 = evaluate("Random Forest", forest, X_train, X_test, y_train, y_test)

importance = pd.Series(forest.feature_importances_, index=X_train.columns)
print("\nWhat the Random Forest relies on most:")
print(importance.sort_values(ascending=False).round(3))

if forest_f1 >= log_f1:
    best_name = "Random Forest"
    best_model = forest
else:
    best_name = "Logistic Regression"
    best_model = log_model

MODELS.mkdir(exist_ok=True)
joblib.dump(best_model, MODELS / "risk_model.joblib")
print(f"\nSaved best model ({best_name}) to models/risk_model.joblib")

best_predictions = best_model.predict(X_test)
missed = X_test[(y_test == 1) & (best_predictions == 0)]
print("\nHigh-cost customers the model missed:", len(missed))
print("How many of them smoke:", int(missed["smoker_yes"].sum()))