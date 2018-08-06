import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.mllib.stat import Statistics
import os
from math import sqrt
import matplotlib.pyplot as pt;
from scipy.stats import norm;

def parse_interaction_with_key(line):
     line_split = line.split(",")
     symbolic_indexs = [1,2,3,41]
     clean_line_split = [item for i , item in enumerate(line_split) if i not in symbolic_indexs]
     return (line_split[41] ,np.array([float(x) for x in clean_line_split]))

conf = SparkConf().setMaster("local").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)


data_file = "instrusion_detector_data.gz"
if(os.path.exists(data_file) == False):
    urllib.request.urlretrieve("http://www.desy.de/~cakir/altan_deneme_data.gz" , data_file)
raw_data = sc.textFile(data_file)

vector_data = raw_data.map(parse_interaction_with_key);
normal_labeled_vector_data = vector_data.filter(lambda x :  x[0] =="normal.")
guess_password_label_labeled_vector_data = vector_data.filter(lambda x :  x[0] =="guess_passwd.");

normal_summary  = Statistics.colStats(normal_labeled_vector_data.values());
guess_password_summary = Statistics.colStats(guess_password_label_labeled_vector_data.values());




print ("Preliminary Statistics for label: {}".format("normal"))
print ("Mean: {}".format(normal_summary.mean()[0],3))
print ("St. deviation: {}".format(round(sqrt(normal_summary.variance()[0]),3)))
print ("Max value: {}".format(round(normal_summary.max()[0],3)))
print ("Min value: {}".format(round(normal_summary.min()[0],3)))
print ("Total value count: {}".format(normal_summary.count()))
print ("Number of non-zero values: {}".format(normal_summary.numNonzeros()[0]))

print ("Preliminary Statistics for label: {}".format("guess_passward"))
print ("Mean: {}".format(guess_password_summary.mean()[0],3))
print ("St. deviation: {}".format(round(sqrt(guess_password_summary.variance()[0]),3)))
print ("Max value: {}".format(round(guess_password_summary.max()[0],3)))
print ("Min value: {}".format(round(guess_password_summary.min()[0],3)))
print ("Total value count: {}".format(guess_password_summary.count()))
print ("Number of non-zero values: {}".format(guess_password_summary.numNonzeros()[0]))

normal_duration_values = normal_labeled_vector_data.values().map(lambda x : x[0]).collect()
guess_pasword_duration_values = guess_password_label_labeled_vector_data.values().map(lambda x : x[0]).collect();

#pt.plot(normal_duration_values,norm.pdf(normal_duration_values,np.mean(normal_duration_values),np.std(normal_duration_values)),"-o")
pt.plot(guess_pasword_duration_values,norm.pdf(guess_pasword_duration_values,np.mean(guess_pasword_duration_values),np.std(guess_pasword_duration_values)),"-o")
#pt.hist(normal_duration_values,50);

pt.show()

