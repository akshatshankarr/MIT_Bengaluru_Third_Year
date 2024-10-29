import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2 * X.squeeze() + 1 + np.random.randn(100) * 2  # y = 2x + 1 with noise


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()           # Ordinary Least Squares (No regularization)
ridge = Ridge(alpha=1.0)          # L2 Regularization (Ridge)
lasso = Lasso(alpha=0.1)          # L1 Regularization (Lasso)

lr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

print(f'MSE for Linear Regression (No Regularization): {mse_lr:.2f}')
print(f'MSE for Ridge Regression (L2 Regularization): {mse_ridge:.2f}')
print(f'MSE for Lasso Regression (L1 Regularization): {mse_lasso:.2f}')

#Visualising.

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='black', label='Data')

plt.plot(X_test, y_pred_lr, color='blue', label='Linear Regression (No Regularization)')
plt.plot(X_test, y_pred_ridge, color='green', label='Ridge Regression (L2 Regularization)')
plt.plot(X_test, y_pred_lasso, color='red', label='Lasso Regression (L1 Regularization)')

plt.title('Effect of Regularization on Regression Models')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()