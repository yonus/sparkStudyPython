import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.stat import Statistics;
import os;

def parse_interaction_with_key(line):
     line_split = line.split(",")
     symbolic_indexs = [1,2,3,41]
     clean_line_split = [item for i , item in enumerate(line_split) if i not in symbolic_indexs]
     return (line[41] ,np.array([float(x) for x in clean_line_split]))

conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

data_file = "instrusion_detector_data.gz"
if(os.path.exists(data_file) == False):
 urllib.request.urlretrieve("http://www.desy.de/~cakir/altan_deneme_data.gz" , "instrusion_detector_data.gz")

raw_data = sc.textFile(data_file)

vector_data = raw_data.map(parse_interaction_with_key);
labeled_vector_data = vector_data.filter(lambda x : x[0] == ".gues_password");

summary = Statistics.colStats(labeled_vector_data.values());
print("Duration mean :", summary.mean()[0])
print("Duration max :" ,summary.max()[0])
#print(raw_data.take(5))
#print(vector_data.take(5))    



