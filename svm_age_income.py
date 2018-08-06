import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#clustering data generation
def createClusteredData(N,k):
    pointsPerCluster=float(N)/k
    X=[] #will contain income X[0] and age X[1]
    y=[] #will number each cluster
    for i in range (k):
        incomeCentroid=np.random.uniform(20000,200000)
        ageCentroid=np.random.uniform(20,70)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000), np.random.normal(ageCentroid, 2)])
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    return X, y

def create_meshgrid(x, y):
    x_min, x_max = x.min() , x.max() 
    y_min, y_max = y.min() , y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 10000),
                         np.arange(y_min, y_max, 2))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    #Plot the decision boundaries for a classifier with mesgrid outputs for given axes.
    """
    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

#Figure our samples  with 5 cluster and 200 points
(X,y)=createClusteredData(100,4)
age, income = X[:, 0], X[:, 1]
plt.scatter(age, income, c=y.astype(np.float)) 
plt.show()

#list of three classifier with different kernel
C = 1.0
kernels = ['linear', 'poly', 'rbf','sigmoid'];
classifiers  = [svm.SVC(kernel='linear',C=C), svm.SVC(kernel='rbf', gamma=0.7,C=C),svm.SVC(kernel='poly', degree=3, C=C),svm.SVC(kernel='sigmoid', C=C)]

titles = ['SVC with linear kernel)',
          'SVC with RBF kernel',
          'SVC with p',
          'sigmoid' ]
print("sdddfdfdfdfdfdfdfd")
models = [clf.fit(X,y)  for clf in classifiers];
age_xx, income_yy = create_meshgrid(age, income)
print("dddfdfdfdfdfdfdfd")
fig, subplots = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plotData = zip(models , titles, kernels,subplots.ravel())
print("fdfdfdfdfdfdfd")
for model , title ,kernel , ax in plotData :
    plot_contours(ax, model, age_xx, income_yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(age, income, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(age_xx.min(), age_xx.max())
    ax.set_ylim(income_yy.min(), income_yy.max())
    ax.set_xlabel('Income')
    ax.set_ylabel('Age')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    
plt.show()

     







