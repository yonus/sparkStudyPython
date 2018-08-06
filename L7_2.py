import numpy as np
from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from numpy import array, random
from sklearn.preprocessing import scale
from pyspark.mllib.clustering import KMeans
from math import sqrt
conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

def binary(YN):
    if YN == "Y":
        return 1
    else:
        return 0

def mapEducation(degree):
     if degree == "MS":
         return 1
     elif degree == "BS":
         return 2
     elif degree == "PhD":
         return 3
     else:
        return 0

def createLabelPoint(fields):
      yearsofexperience =  int(fields[0])
      employed = binary(fields[1])
      previousEmployers = int(fields[2])
      educationLevel = mapEducation(fields[3]) 
      topTire = binary(fields[4])
      internal = binary(fields[5])
      hired = binary(fields[6])
      return LabeledPoint(hired , np.array([yearsofexperience,employed,previousEmployers,educationLevel,topTire,internal]))     



rawData = sc.textFile("data/PastHires.csv")
header = rawData.first()
rawData = rawData.filter(lambda x: x != header)
csvData = rawData.map(lambda x : x.split(","))
trainingData = csvData.map(createLabelPoint)
testCandidates = [np.array([10,1,3,1,0,0])]
testData = sc.parallelize(testCandidates)

model = DecisionTree.trainClassifier(trainingData,numClasses = 2 ,categoricalFeaturesInfo = {1:2,3:4,4:2,5:2})
prediction = model.predict(testData)
result = prediction.collect();


def createClusterData(N , k):
   random.seed(10)
   pointCluster  = float(N)/k;
   X = []
   for i in range(k):
       incomeCentroid = random.uniform(20000.0 , 200000.0);
       ageCetroid = random.uniform(20.0, 70.0)
       for j in range (int(pointCluster)):
         X.append([random.normal(incomeCentroid,10000.0) , random.normal(ageCetroid,2.0)])
   X = array(X);
   return X;

K = 5;


clusterData = sc.parallelize(scale(createClusterData(100,K)))
clusters = KMeans.train(clusterData ,K, maxIterations=10 , initializationMode="random")

resultRDD = clusterData.map(lambda point : clusters.predict(point)).cache()

print("Count by values")
counts = resultRDD.countByValue()
print(counts)

print("Cluster assigment")
result = resultRDD.collect()
print(resultRDD)

def error(point):
    center  = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point -center)]))
 