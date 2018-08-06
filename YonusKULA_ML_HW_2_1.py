import numpy as np
import matplotlib.pyplot as plt

#random first 100 number
x = np.random.normal(10.0,3.0, 100);


# random 100 number from x with adding noise
y  =  x * (4- np.random.randint(1,4,100));

print(np.cov(x,y))

print(np.corrcoef(x,y))

plt.scatter(x,y)
plt.show()

