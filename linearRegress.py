import numpy as np 
from sklearn.linear_model import LinearRegression

X = 2*np.random.rand(100,1)
y = 4 + 3*X + np.random.randn(100,1)

linReg = LinearRegression()
linReg.fit(X,y)

print("Intercept: ", linReg.intercept_)
print("Coefficients: ", linReg.coef_)

X_new = np.array([[0], [2]])
y_pred = linReg.predict(X_new)

print('Predicted vals: ', y_pred)