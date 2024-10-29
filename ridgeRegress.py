from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

krr = KernelRidge(kernel='rbf', alpha=0.5, gamma=0.1)
krr.fit(X_train, y_train)

y_pred = krr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Ridge mse: ", mse*100)