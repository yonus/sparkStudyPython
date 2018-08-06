import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

def parsePoint(line):
    values = [float(x) for x in line.replace(","," ").split(' ')]
    return LabeledPoint(values[0] , values[1:])




data = sc.textFile("data/lpsa.data")
parsedData = data.map(parsePoint)

model = LinearRegressionWithSGD.train(parsedData,iterations=100, step=0.00001)

 # Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
mSE = valuesAndPreds .map(lambda vp: (vp[0] - vp[1])**2) .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(mSE))

# Save and load model
model.save(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
sameModel = LinearRegressionModel.load(sc, "target/tmp/pythonLinearRegressionWithSGDModel")
