import numpy as np;
import matplotlib.pyplot as plt;

N = 50;
X = np.random.rand(N)
Y = np.random.rand(N)
colors = np.random.rand(N)

area = (colors*30)**2;

plt.scatter(X,Y,s=area,c=colors, alpha=0.5)
plt.show();