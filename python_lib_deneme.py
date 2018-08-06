#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:10:18 2018
@author: ayse
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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

#Create 4 clusters with 25 data points per cluster
(X,y)=createClusteredData(100,4)

plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y.astype(np.float)) 
plt.show()

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          
          svm.SVC(kernel='rbf', gamma=0.7, C=C))
models = (clf.fit(X, y) for clf in models)
# title for the plots
titles = ('SVC with linear kernel',
          'SVC with RBF kernel')
# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1] #split as income and age

def make_meshgrid(x, y):
    #h: stepsize for meshgrid, optional
    x_min, x_max = x.min() , x.max() 
    y_min, y_max = y.min() , y.max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 10000),
                         np.arange(y_min, y_max, 2))
    return xx, yy
xx, yy = make_meshgrid(X0, X1)

def plot_contours(ax, clf, xx, yy, **params):
    #Plot the decision boundaries for a classifier.
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


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Income')
    ax.set_ylabel('Age')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show()
