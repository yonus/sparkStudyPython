from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.sql.types import Row
from pyspark.sql.functions import col, approxCountDistinct ,countDistinct
#import pandas as pd;

from pyspark.ml.feature import OneHotEncoderEstimator,StringIndexer,VectorAssembler,Imputer
from pyspark.ml.pipeline  import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression,DecisionTreeClassifier ,RandomForestClassifier,GBTClassifier
from pyspark.sql.functions import Column 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
integerColums = range (1,14)
categoricalColumns = range(14,40)
numericColumnNames = []
categoricalColumnsNames = [];
plotEanbled = False;
STRING_INDEXER_OUT_SUFFIX = "_out_index"
ONE_HOT_ENCODER_OUT_SUFFIX = "_out_ohe"

def lineToRow(line , labelColunmName = "label" , numericColumnNames = [] ,categoricalColumnNames = []):
     columns = line.split("\t")
     dict = {};
     dict[labelColunmName] = float(columns[0]);
     for i ,columnName in zip(integerColums, numericColumnNames):
        if len(columns[i]) > 0:
           dict[columnName] = float(columns[i])
        else:
           dict[columnName] = None;
         
     for c ,columnName in zip(categoricalColumns,categoricalColumnNames):
         if columns[c] and columns[c].strip() :
           dict [columnName] = columns[c].strip()
         else:  
           dict [columnName] = None
     return Row(**dict);

def getColumnStructure():
    numericColumnNames = [];
    categoricalColumnNames = []
    for i in integerColums:
        columnName = "I"+str(i); 
        numericColumnNames.append(columnName);    
    for c in categoricalColumns:
         columnName = "c"+str(c-13) 
         categoricalColumnNames.append(columnName)
    return numericColumnNames , categoricalColumnNames     

def readDataFrame(path):
   return sqlContext.read.option("header", "false").option("inferSchema", "true").option("delimiter", "\t").csv(path)



#click = readDataFrame("s3a://wer-display-ads/day_0_1000.csv");
#click.describe()

def getColumnUniqueCounts(clickDF):
    """
    Parameters
    ---------- 
    clickDF : class Dataframe
    """ 
    distvals = clickDF.agg(*(countDistinct(col(c)).alias(c) for c in clickDF.columns))
    print(type(distvals))
    return distvals

def plotColumnUniqueCounts(distinctCountDF):
   if plotEanbled :
        import matplotlib.pyplot as plt
        firstRow  = distinctCountDF.iloc[0]
        firstRow.plot(kind="bar")  
        plt.show()

def oneHotEncoding(clickDF  , columns):
    """
    ohe = OneHotEncoderEstimator
    """
    
    allStages = [StringIndexer(inputCol=column, outputCol=column+STRING_INDEXER_OUT_SUFFIX).setHandleInvalid("skip") for column in columns]
    oneHotEncodeInputOutputNames = [(column+STRING_INDEXER_OUT_SUFFIX , column+ONE_HOT_ENCODER_OUT_SUFFIX) for column in columns]
    oneHotEncodeInputOutputNames = list(zip(*oneHotEncodeInputOutputNames))
    ohe = OneHotEncoderEstimator(inputCols=oneHotEncodeInputOutputNames[0] , outputCols=oneHotEncodeInputOutputNames[1])
    allStages.append(ohe);
    pipeline = Pipeline(stages=allStages)
    clickDF =  pipeline.fit(clickDF).transform(clickDF)
    deletedColumns = list(oneHotEncodeInputOutputNames[0])+columns; 
    return clickDF;

def impute(clickDF  , numericColumnNames=[]):
    outputColumNames = [columnName+"_out_imputer" for columnName in numericColumnNames] 
    #imputer =  Imputer().setInputCols(numericColumnNames).setOutputCols(outputColumNames)  
    imputer =  Imputer().setInputCols(numericColumnNames).setOutputCols(outputColumNames).setMissingValue(0)
    return imputer.fit(clickDF).transform(clickDF) , outputColumNames;


def dataToVectorForLinear(clickDF , categoricalColumnsNames , numericColumnNames):
  print ("=====One hot encoding=======")
  clickDF = oneHotEncoding(clickDF,categoricalColumnsNames)
  print ("===== Imputing=======")
  clickDF , imputedColumnNames = impute(clickDF,numericColumnNames)
  all_feature_columns = imputedColumnNames + [columnName + ONE_HOT_ENCODER_OUT_SUFFIX for columnName in categoricalColumnsNames];
  print ("===== Assambler =======")
  feature_assembler = VectorAssembler(inputCols=all_feature_columns,outputCol="features")
  return feature_assembler.transform(clickDF);

def dataToVectorForTree(clickDF,categoricalColumnsNames , numericColumnNames):
  print ("===== Imputing=======") 
  clickDF , imputedColumnNames = impute(clickDF,numericColumnNames)
  
  print ("===== String Indexer=======") 
  
  allStages = [StringIndexer(inputCol=column, outputCol=column+STRING_INDEXER_OUT_SUFFIX).setHandleInvalid("skip") for column in categoricalColumnsNames]
  stringIndexderColumnsNames = [(column+STRING_INDEXER_OUT_SUFFIX , column+ONE_HOT_ENCODER_OUT_SUFFIX) for column in categoricalColumnsNames] 
  stringIndexderColumnsNames = list(zip(*stringIndexderColumnsNames))
  pipeline = Pipeline(stages=allStages)
  clickDF =  pipeline.fit(clickDF).transform(clickDF)
  all_feature_columns = imputedColumnNames + list(stringIndexderColumnsNames[0]);
  print ("===== Assambler =======")
  feature_assembler = VectorAssembler(inputCols=all_feature_columns,outputCol="features")
  return feature_assembler.transform(clickDF);


def evaluateLogisticRegression(trainDF , testDF):
  for reg in [0.1 ,1 ,10 ,100] :
    mlr = LogisticRegression( maxIter=10, regParam=reg, elasticNetParam=0.8, family="multinomial")
    model = mlr.fit(trainDF);
    predictions = model.transform(testDF).cache()
    print("Logistic Regrassion for reg {}".format(reg))
    printevaluatation(model,predictions)

def evaluateDecisionTree(trainDF ,testDF):
    for maxDepth in [10,15]:
      classifier = DecisionTreeClassifier(maxDepth=maxDepth ,maxBins=)
      model = classifier.fit(trainDF)
      
      predictions = model.transform(testDF)
      print("Decision Tree evaluation with maxtDepth : {}".format(maxDepth))
      printevaluatation(model,predictions)

def evaluateRandomForest(trainDF ,testDF):
    for numTree in [5,15,25]:
      classifier = RandomForestClassifier(numTrees = numTree);
      model = classifier.fit(trainDF)
      predictions = model.transform(testDF)
      print("Random Forest with numThree:{}".format(numTree))
      printevaluatation(model,predictions)
def evaluateGradientBoostTree(trainDF ,testDF):
    for stepsize in [0.01 ,0.1 ,1]:
        classifier = GBTClassifier(stepSize=stepsize)
        model = classifier.fit(trainDF)
        predictions = model.transform(testDF)
        print("Gradient Boost Tree with stepsize : {}".format(stepsize))
        printevaluatation(model,predictions)

def evaluateSVM(trainDF ,testDF):
    for reg in [0.01,0.1,1,10]:
       classifier = LinearSVC(regParam=reg)
       model = classifier.fit(trainDF)
       predictions = model.transform(testDF)
       print("SVM with reg: {}".format(reg))
       printevaluatation(model, predictions)

def printevaluatation(model , predictions):
    evalator_auc  = MulticlassClassificationEvaluator(labelCol="label" ,predictionCol="prediction")
    evalator_accuracy = MulticlassClassificationEvaluator(labelCol="label" ,predictionCol="prediction", metricName="accuracy")
    if(hasattr(model,"summary")):
        print("acurracy {}".format(model.summary.accuracy))
        print("roc {}".format(model.summary.roc))
        print("precision {}".format(model.summary.precisionByLabel))
        print("recall {}". format(model.summary.recallByLabel))
        print("under roc {}".format(model.summary.areaUnderROC))
    print("test error {}".format(1- evalator_accuracy.evaluate(predictions)))
    print("test uac {}".format(evalator_auc.evaluate(predictions)))



def main():
  #clickRDD = sc.textFile("s3a://wer-display-ads/day_0_1000.csv"); 
  clickRDD = sc.textFile("data/day_0_1000.csv");
  numericColumnNames , categoricalColumnsNames = getColumnStructure();
  #clickDF = clickDF.cache()
  
  clickRDDRows = clickRDD.map(lambda line : lineToRow(line,labelColunmName="label" ,numericColumnNames=numericColumnNames , categoricalColumnNames=categoricalColumnsNames));
  clickDF = sqlContext.createDataFrame(clickRDDRows)
  #distinctCountDF = getColumnUniqueCounts(clickDF);
  #distinctCountDF = distinctCountDF.toPandas()
  #print(distinctCountDF.head())
  #plotColumnUniqueCounts(distinctCountDF)
  
  """
  clickDFLinear = dataToVectorForLinear(clickDF,categoricalColumnsNames,numericColumnNames);
  trainDFLinear , testDFLinear = clickDFLinear.randomSplit([0.8,0.2])
  trainDFLinear.cache()
  testDFLinear.cache()
  evaluateLogisticRegression(trainDFLinear, testDFLinear)
  evaluateSVM(trainDFLinear, testDFLinear)
  """
  clickDFTree = dataToVectorForTree(clickDF,categoricalColumnsNames,numericColumnNames)
  trainDFTree , testDFTree = clickDFTree.randomSplit([0.8,0.2])
  evaluateDecisionTree(trainDFTree , testDFTree)
  evaluateRandomForest(trainDFTree , testDFTree)
  evaluateGradientBoostTree(trainDFTree , testDFTree)
  

if __name__ == "__main__":
    main()
