from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#initialize lr systems w/ L1 or L2 penalty incurring systems w/ liblinear solver to minimize cost fn
#liblinear is made for small/medium scale datasets; uses one versus rest approach.
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
lr_l2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)

lr_l1.fit(X_train, y_train)
lr_l2.fit(X_train, y_train)

y_pred_l1 = lr_l1.predict(X_test)
y_pred_l2 = lr_l2.predict(X_test)

accuracy_l1 = accuracy_score(y_test, y_pred_l1)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)

print('L1 accuracy is: ', accuracy_l1)
print('L2 accuracy is: ', accuracy_l2)

conf_l1 = confusion_matrix(y_test, y_pred_l1)
conf_l2 = confusion_matrix(y_test, y_pred_l2)

print('L1 conf matrix is: ', conf_l1)
print('L2 conf matrix is: ', conf_l2)