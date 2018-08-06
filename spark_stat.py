import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.stat import Statistics

conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)


   

mat = sc.parallelize(
    [np.array([1.0, 10.0, 100.0]), np.array([2.0, 20.0, 200.0]), np.array([3.0, 30.0, 300.0])]
)   

 

# Compute column summary statistics.
summary = Statistics.colStats(mat)
print(summary.mean())  # a dense vector containing the mean value for each column
print("sdsdsffsfsffsfs", summary.variance())  # column-wise variance
print("mmmmmmmmmmmmmmmmmm " ,summary.numNonzeros())  # number of nonzeros in each column

sc.stop();


