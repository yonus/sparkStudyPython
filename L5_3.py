from pyspark import SparkContext,SparkConf
from pyspark.mllib.tree import DecisionTree , DecisionTreeModel
from pyspark.mllib.util import MLUtils

conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)

data =  MLUtils.loadLibSVMFile(sc,"data/sample_libsvm_data.txt");

train, test = data.randomSplit([0.7,0.3])

model = DecisionTree.trainClassifier(train);

model.predict(test.map(lambda x : x.features))
