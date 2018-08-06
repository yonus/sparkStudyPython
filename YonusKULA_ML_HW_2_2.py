import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.mllib.stat import Statistics
from pyspark.sql.functions import length
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import os
conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)


data_file = "Musical_Instruments_5.json"
if(os.path.exists(data_file) == False):
    urllib.request.urlretrieve("https://s3.eu-central-1.amazonaws.com/wer-amazon-review/Musical_Instruments_5.json" , data_file)

sql = SQLContext(sc)
reviews =  sql.read.json(data_file)
reviews = reviews.withColumn('reviewLength', length('reviewText'))
#reviews_x = reviews.select(["reviewLength" ,"overall" ] )
/home/pasa/sparkdevelopment/YonusKULA_ML_HW_2_2.py

reviews_x = assembler.transform(reviews)

#pearson correaltion
r1 = Correlation.corr(reviews_x,"feature").collect()[0][0];
print ("Pearson Correlation Matrix: \n" + str(r1));

r2 = Correlation.corr(reviews_x, "feature", "spearman").collect()[0][0]
print ("Spearman Correlation Matrix: \n" + str(r2));




