import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

iris = load_wine()
X, y = iris.data, iris.target

iris_df = pd.DataFrame(data= X, columns= iris.feature_names)
print('Iris dataset sample: ', iris_df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# initialize rfc with gini impurity categorization i.e. P(incorrectly classifying test case according to class label distribution in tree node)
# n_estimators allow for 100 trees in the rfc system
rf = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('Accuracy Score: ', accuracy*100)
print('Classification Report: ')
print(report)
