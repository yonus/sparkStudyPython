from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.mllib.feature import HashingTF

conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

rawData = sc.textFile("data/subset-small.csv")
fields = rawData.map(lamda x : x.split("\t"))
documents = fields.map(lambda x : x[3].split(" "))


documentsNames = fields.map(lambda x : x[1])

