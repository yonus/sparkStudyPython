from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl;
from itertools import cycle
iris = load_iris()

numSamples , numFeature = iris.data.shape

X = iris.data

pca = PCA(n_components=2 ,whiten=True).fit(X)
X_PCA = pca.transform(X)

print(pca.components_)
print(pca.explained_variance_ratio_)

colors = cycle('rgb')
targets_id =  range(len(iris.target_names))
pl.figure();

for i , c , label in zip(targets_id,cycle,iris.target_names):
    
