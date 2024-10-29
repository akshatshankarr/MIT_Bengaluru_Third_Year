from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dc = DecisionTreeClassifier()
dc.fit(X_train, y_train)

y_pred = dc.predict(X_test)

accuracy = dc.score(X_test, y_test)
accuracy_metric = accuracy_score(y_test, y_pred)

print('Metric: ', accuracy_metric)
print('Accuracy: ', accuracy)