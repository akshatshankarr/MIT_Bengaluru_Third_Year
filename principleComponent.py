from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA 
import pandas as pd 
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y= iris.target

X_std = (X-X.mean())/X.std()

pca = PCA(n_components=2)
pca.fit(X_std)

X_pca = pca.transform(X_std)

principal_components_df = pd.DataFrame(X_pca, columns=['PC1','PC2'])
final_df = pd.concat([principal_components_df, pd.DataFrame(y, columns = ['target'])], axis=1)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)

targets = [0,1,2]
colors = ['r','g','b']

for target, color in zip(targets, colors):
    indices = final_df['target']==target
    ax.scatter(final_df.loc[indices, 'PC1'], final_df.loc[indices, 'PC2'], c=color, s=50)

plt.show()