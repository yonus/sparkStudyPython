import numpy as np;
from pyspark import SparkContext,SparkConf
from pyspark.mllib.clustering import KMeans , KMeansModel , BisectingKMeans, BisectingKMeansModel
from math import sqrt

conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

data = sc.textFile("data/kmeans_data.txt")
parsedData = data.map(lambda line : np.array([float(x) for x in line.split(' ')]))

kmeans_model = KMeans.train(parsedData,2,maxIterations=100)


def error(point):
    center = kmeans_model.centers[kmeans_model.predict(point)]
    return sqrt(sum([x**2 for x in (point- center)]))

WSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y : x + y);
print("WSSSE is : ",WSSE)
print("K-meanns cost : ", kmeans_model.computeCost(parsedData))

bisectionKmeansModel = BisectingKMeans.train(parsedData,2,maxIterations=5)
cost = bisectionKmeansModel.computeCost(parsedData);
print("Bisection cost : ", bisectionKmeansModel.computeCost(parsedData))

