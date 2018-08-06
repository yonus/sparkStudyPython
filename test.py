import urllib
import urllib.request 
import numpy as np
import matplotlib.pyplot as plt
from pyspark.mllib.stat import Statistics 
#from pyspark.mllib.stat import Correlation
from pyspark.ml.linalg import Vectors
from math import sqrt 
from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("ML_hw2_1")
sc = SparkContext(conf = conf)

f = urllib.request.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", "housing.data")
data_file = "./housing.data"
data = sc.textFile(data_file)

def  toVector(data):
    vectorized_key_list = [5,13]
    return np.array([float(data.values[key]) for key in vectorized_key_list])  

vector_data = data.map(toVector)
summary = Statistics.colStats(vector_data)

#Statistics for Median value of owner-occupied homes in $1000's
print ("Statistics for Median Value of Homes:")
print (" Mean: {}".format(round(summary.mean()[1],0)))
print (" St. deviation: {}".format(round(sqrt(summary.variance()[1]),0)))
print (" Max value: {}".format(round(summary.max()[1],0)))
print (" Min value: {}".format(round(summary.min()[1],0)))
print (" Total value count: {}".format(summary.count()))
print (" Number of non-zero values: {}".format(summary.numNonzeros()[1]))

#Statistics for Average number of rooms per dwelling
print ("Statistics for Average Number of Rooms per Dwelling:")
print (" Mean: {}".format(round(summary.mean()[0],0)))
print (" St. deviation: {}".format(round(sqrt(summary.variance()[0]),0)))
print (" Max value: {}".format(round(summary.max()[0],0)))
print (" Min value: {}".format(round(summary.min()[0],0)))
print (" Total value count: {}".format(summary.count()))
print (" Number of non-zero values: {}".format(summary.numNonzeros()[0]))

MedianVal=vector_data.map(lambda y: round(float(y[1]),0)) 
MedianVal=MedianVal.collect()

AvgRoom=vector_data.map(lambda y: round(float(y[0]),0)) 
AvgRoom=AvgRoom.collect()

#Scatter Plot
plt.scatter(MedianVal, AvgRoom)
plt.title("Scatter Plot for Median Value of Homes and Average Number of Rooms")
plt.xlabel("Median Value of Homes (in $1000's)")
plt.ylabel("Average Number of Rooms per Dwelling")
plt.show()

# Defining mean function
def de_mean(x):
	xmean=np.mean(x)
	return [i - xmean for i in x]

#Defining covariance function
def covariance(x,y):
	n=len(x)
	return np.dot(de_mean(x), de_mean(y))/ (n-1)

#Defining correlation function
def correlation(x,y):
    stddevx=x.std()
    stddevy=y.std()
    return covariance(x,y) / stddevx /stddevy

print(covariance(MedianVal,AvgRoom))
print(Correlation.corr(MedianVal, AvgRoom))
print(Correlation.corr(MedianVal, AvgRoom, "spearman"))