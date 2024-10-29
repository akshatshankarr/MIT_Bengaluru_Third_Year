import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import seaborn as sns

def plotResult(X_test, y_test, y_pred, samples=10):
    plt.figure(figsize=(15,2))
    for i in range(samples):
        ax = plt.subplot(1, samples, i+1)
        plt.imshow(X_test[i].reshape(8,8), cmap='gray')
        plt.title(f'Predict: {y_pred[i]}\n Actual: {y_test[i]}')
        plt.axis(False)
    plt.show()

def plotSample(X, y, samples= 10):
    plt.figure(figsize=(8,2))
    for i in range(samples):
        ax = plt.subplot(1, samples, i+1)
        plt.imshow(X[i].reshape(8,8), cmap='gray')
        plt.title(y[i])
        plt.axis(False)
    plt.suptitle('Sample images')
    plt.show()

digits = load_digits()
X, y = digits.data, digits.target

#Display sample data
plotSample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9, stratify=y)

model = SVC(kernel='linear', C=1.0) #Replacable with any other classification model.
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

cv = StratifiedKFold()
cv_score = cross_val_score(model, X_train, y_train, cv= cv)
print(f'{cv_score.mean():.2f}+-{cv_score.std():.2f}')

#Plotting the cross val scores
plt.figure(figsize=(8,6))
plt.plot(range(1, len(cv_score)+1), cv_score, marker='o')
plt.ylim(0.8,1.0)
plt.grid(True)
plt.show()

conf_matrix = confusion_matrix(y_test, y_pred)

#Display heatmap
sns.heatmap(conf_matrix, annot=True, cmap='gray')
plt.show()

#Display confusion matrix
display = ConfusionMatrixDisplay(conf_matrix, display_labels=digits.target_names)
display.plot(cmap='gray')
plt.show()

#Displaying predictions
plotResult(X_test, y_test, y_pred)

cv_score = cross_val_score(model,X_train, y_train, cv=cv)