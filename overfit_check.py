"""Show overfitting by removing the depth limit on the random forest."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from prepare import get_train_test

X_train, X_test, y_train, y_test = get_train_test()

deep_forest = RandomForestClassifier(n_estimators=200, random_state=42)
deep_forest.fit(X_train, y_train)

print("Deep forest train accuracy:", round(accuracy_score(y_train, deep_forest.predict(X_train)), 3))
print("Deep forest test accuracy: ", round(accuracy_score(y_test, deep_forest.predict(X_test)), 3))
