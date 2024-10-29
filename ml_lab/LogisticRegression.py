from sklearn.datasets import make_classification, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
iris = load_wine()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

lr = LogisticRegression(random_state=42)
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy is: {:.2f}%".format(accuracy*100))
