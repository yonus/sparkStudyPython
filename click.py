from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.sql.types import Row

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
integerColums = range (1,14)
categoricalColumns = range(14,40)


def lineToRow(line):
     columns = line.split("\t")
     dict = {};
     dict["label"] = columns[0];
     for i in integerColums:
         if len(columns[i]) > 0:
           dict["I"+str(i)] = int(columns[i])
         else:
           dict["I"+str(i)] = None;
         
     for c in categoricalColumns:
         dict ["c"+str(c-13)] = columns[c]
         
         
     return Row(**dict);


def readDataFrame(path):
   return sqlContext.read.option("header", "false").option("inferSchema", "true").option("delimiter", "\t").csv(path)



click = readDataFrame("s3a://wer-display-ads/day_0_1000.csv");
#click.describe()

clickRDD = sc.textFile("s3a://wer-display-ads/day_0_1000.csv"); 
clickRDDRows = clickRDD.map(lineToRow);
clickDF = sqlContext.createDataFrame(clickRDDRows)
clickDF.describe()
#clickDF.show()
