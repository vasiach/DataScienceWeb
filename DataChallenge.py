import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from feature_extr import feature_extr

# read train and test files
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test = pd.read_csv("data/X_test.csv")

X_train = feature_extr(X_train)


X_example_train, X_example_test, y_example_train, y_example_test = train_test_split(X_train, y_train, test_size=0.33)
logreg = LogisticRegression()
logreg.fit(X_example_train, y_example_train["category"])
y_pred = logreg.predict(X_example_test)

print("logreg", accuracy_score(y_example_test["category"], y_pred))