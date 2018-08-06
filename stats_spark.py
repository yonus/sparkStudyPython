
import numpy as np
import os;
from pyspark import SparkConf, SparkContext
from pyspark.mllib.stat import Statistics

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext.getOrCreate();


   

mat = sc.parallelize(
    [np.array([1.0, 10.0, 100.0]), np.array([2.0, 20.0, 200.0]), np.array([3.0, 30.0, 300.0])]
)   

 

# Compute column summary statistics.
summary = Statistics.colStats(mat)
print(summary.mean())