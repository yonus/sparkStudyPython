import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt

np.random.seed(2)

NumberOfItem = np.random.normal(3.0,1.0,100)
purchasedAmount = np.random.normal(50.0 , 30.0 , 100) / NumberOfItem;

plt.scatter(NumberOfItem,purchasedAmount)
