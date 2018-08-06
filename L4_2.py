import numpy as np;
import random;

from sklearn.cluster import KMeans;
from sklearn.preprocessing import scale;
import matplotlib.pyplot as plt;
def createClusterData(N , k):
   random.seed(10)
   pointCluster  = float(N)/k;
   X = []
   for i in range(k):
       incomeCentroid = random.uniform(20000.0 , 200000.0);
       ageCetroid = random.uniform(20.0, 70.0)
       for j in range (int(pointCluster)):
         X.append([np.random.normal(incomeCentroid,10000.0) , np.random.normal(ageCetroid,2.0)])
   X = np.array(X);
   return X;


trainData = createClusterData(1000,2);
model = KMeans(5)
scaled_data = scale(trainData);

model = model.fit(scaled_data);

#print(model.labels_)
print(scaled_data)
plt.figure(figsize=(7,8))
plt.scatter(scaled_data[:,0],scaled_data[:,1],c=model.labels_.astype(float))
plt.show()




