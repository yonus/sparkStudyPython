import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.stat import Statistics;
import os.path

# convert  text data in json format to json object dictionary
# also we add review lenght dynamically to use at statistic 
def reviewTextToJson(reviewText):
    reviewJson = eval(reviewText)
    if "reviewText" in  reviewJson :
       reviewJson["reviewLength"] = len(reviewJson["reviewText"])
    else :   
       reviewJson["reviewLength"] = 0
    return reviewJson

# reviewLength and overall values are converted to vector format
def  toVector(reviewJson):
    vectorized_key_list = ["reviewLength" ,"overall"]
    return np.array([float(reviewJson[key]) for key in vectorized_key_list])  


conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)


FILE_NAME = "Musical_Instruments.json"
FILE_URL = "https://s3.eu-central-1.amazonaws.com/wer-amazon-review/Musical_Instruments_5.json";


# if this file don't exist in local , we download to local  
if(os.path.exists(FILE_NAME) == False):
   urllib.request.urlretrieve(url=FILE_URL,filename=FILE_NAME)

reviewsRDD = sc.textFile(FILE_NAME).map(reviewTextToJson)
vector = reviewsRDD.map(toVector)
summary = Statistics.colStats(vector)

print("Review Text Length and overall statistic information:")
print("Mean of Review Text Length : {}".format(summary.mean()[0]))
print("Mean of Overall : {}".format(summary.mean()[1]))

print("Variance of Review Text Length : {}".format(summary.variance()[0]))
print("Variance of Overall : {}".format(summary.variance()[1]))

print("Count of Review Text Length : {}".format(summary.count()))
print("Count of Overall : {}".format(summary.count()))

print("Max of Review Text Length : {}".format(summary.max()[0]))
print("Max of Overall : {}".format(summary.max()[1]))

print("Min of Review Text Length : {}".format(summary.min()[0]))
print("Min of Overall : {}".format(summary.min()[1]))


