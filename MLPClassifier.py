from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

clf = MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))