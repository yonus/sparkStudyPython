from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.ml.regression import LinearRegression

from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)


inputLines =   sc.textFile("dfdfdfdfd.txr");
data = inputLines.map(lambda x : x.split(",")).map(lambda x : (float(x[0]) ,Vectors.dense(float(x[1]))));
colnames = ["label","features"]
df = data.toDF(colnames)

trainTest = df.randomSplit([0.5,0.5])
traninDF = trainTest[0]
testDF = trainTest[1];

lir = LinearRegression(maxIter=10 ,regParam=0.3, elasticNetParam = 0.8)
model = lir.fit(traninDF)

fullPredictions = model.transform(testDF).cache();






