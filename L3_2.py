import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



numberOfItem = np.random.normal(3.0 ,1.0 ,1000)
purchasedAmount = 100 -  (numberOfItem  + np.random.normal(0,0.1 , 1000))
purchasedAmount2 = np.random.normal(50.0,10.0,1000)/numberOfItem


#scatter(numberOfItem, purchasedAmount)
slope , intercept , r_value , p_value , std_err = stats.linregress(numberOfItem,purchasedAmount2);
def predict (x):
     return slope*x + intercept;


fitline = predict(numberOfItem)

plt.scatter(numberOfItem , purchasedAmount2)
plt.plot(numberOfItem,fitline,c="r")
plt.show()

x = np.array(numberOfItem);
y = np.array(purchasedAmount)

p4 = np.poly1d(np.polyfit(x,y,4))

xp = np.linspace(0,7,100)
plt.scatter(x,y)
plt.plot(xp ,p4(xp) , c="r")
plt.show();

