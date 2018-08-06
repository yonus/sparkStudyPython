import urllib.request
import numpy as np
from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.sql.functions import Column as col
from pyspark.ml.regression import LinearRegression
from  pyspark.ml.feature import OneHotEncoder,StringIndexer,StandardScaler,VectorAssembler
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt;
conf = SparkConf().setMaster("local[*]").setAppName("Spark-Stat")
sc = SparkContext(conf=conf)
sqlContext  = SQLContext(sc)

housingDF = sqlContext.read.csv("housing.csv",header=True)
housingDF = housingDF.na.drop();

labelColumn = "median_house_value"
categorical_columns = ["ocean_proximity"]
numericalColumns = [x for x in housingDF.columns if x not in categorical_columns and x != labelColumn]

housingDF = housingDF.withColumn("median_house_value",housingDF["median_house_value"].cast("float"))
for x in numericalColumns:
  housingDF = housingDF.withColumn(x,housingDF[x].cast("float"))
    
housingDF.printSchema();  


#categorical variables transform

categorical_string_indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categorical_columns]

catergorical_encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),outputCol="{0}_encoded".format(indexer.getOutputCol())) for indexer in categorical_string_indexers]
categorical_encoded_columns = [x.getOutputCol() for x in catergorical_encoders];

numeric_columns_assembler = VectorAssembler(inputCols=numericalColumns, outputCol="numerical_features");
#housingDF = numeric_columns_assembler.transform(housingDF)

scaler = StandardScaler(inputCol="numerical_features",outputCol="numerical_scaled_features")
#housingDF = scaler.fit(housingDF).transform(housingDF);
all_feature_columns = categorical_encoded_columns + ["numerical_scaled_features"];
feature_assembler = VectorAssembler(inputCols=all_feature_columns,outputCol="final_features")
#
# housingDF = feature_assembler.transform(housingDF);
linear_regression = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,labelCol=labelColumn,featuresCol="final_features")

allStages = categorical_string_indexers + catergorical_encoders + [numeric_columns_assembler,scaler,feature_assembler,linear_regression]  
pipeline = Pipeline(stages=allStages)
trainData ,testData = housingDF.randomSplit([0.8,0.2])
model = pipeline.fit(trainData)
regression_model = model.stages[-1]
summary = regression_model.summary
 
print("Coefficients: %s" % str(regression_model.coefficients))
print("Intercept: %s" % str(regression_model.intercept))

# Summarize the model over the training set and print out some metrics

print("numIterations: %d" % summary.totalIterations)
print("objectiveHistory: %s" % str(summary.objectiveHistory))
#summary.residuals.show()
print("RMSE: %f" % summary.rootMeanSquaredError)
print("r2: %f" % summary.r2)
 
#print(regression_model.summary)
predictions = model.transform(testData)
predictions.select("prediction", labelColumn, "final_features").show(20)

