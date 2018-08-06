import numpy as np;
import matplotlib.pyplot as plot;
from  sklearn import svm , datasets
iris = datasets.load_iris();

x = iris.data[:,:2];
y = iris.target

def plotPredictions(clf):
    xx,yy  = np.meshgrid(np.arange(0,250000,10), np.arange(10,70,0.5))
    z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    plot.figure(figsize = (8,6))
    z = z.reshape(xx.shape)
    plot.contourf(xx,yy, z , cmap = plot.cm.Paired,alpha=0.8)
    plot.scatter(x[:,0], x[:,1] , c=y.astype(np.float))
    plot.show()

C = 1.0
svc = svm.SVC(kernel="linear",C = C).fit(x,y)
plotPredictions(svc)