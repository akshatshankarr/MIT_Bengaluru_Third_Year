from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

iris = datasets.load_wine()

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=9)

est = DecisionTreeClassifier(max_depth=1)
clf = AdaBoostClassifier(estimator=est, n_estimators=50, learning_rate=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
print("Accuracy: {:.2f}%".format(acc*100))
print("F1 Score: {:.2f}%".format(f1*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))