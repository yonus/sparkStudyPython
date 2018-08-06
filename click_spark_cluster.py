from pyspark import SparkConf, SparkContext,SQLContext
from pyspark.sql.types import Row
from pyspark.sql.functions import col, approxCountDistinct ,countDistinct,approx_count_distinct
import pandas as pd;


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
plotEanbled = True;
STRING_INDEXER_OUT_SUFFIX = "_out_index"
ONE_HOT_ENCODER_OUT_SUFFIX = "_out_ohe"

plotDirectoy ="s3a://wer-display-ads/plot/"
benchmarkOutputDirectory = "s3a://wer-display-ads/benchmark_result.csv"

#benchmarkOutputDirectory = "benchmark_result.csv"
#plotDirectoy ="./"


def lineToRow(line , labelColunmName = "label" , numericColumnNames = [] ,categoricalColumnNames = []):
 """
  convert text line to DataFrame Row Object
 """
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
    """
    return column names to used in dataframe
    """
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




def getColumnUniqueCounts(clickDF):
    """
    Parameters
    ---------- 
    clickDF : class Dataframe 
    """ 
    distvals = clickDF.agg(*(approx_count_distinct(col(c)).alias(c) for c in clickDF.columns if str.startswith(c,"c")))
    print(type(distvals))
    return distvals

def plotColumnUniqueCounts(distinctCountDF):
   if plotEanbled :
        import matplotlib
        matplotlib.use('agg',warn=False, force=True)
        import matplotlib.pyplot as plt
        from io import BytesIO;
        firstRow  = distinctCountDF.iloc[0]
        firstRow.plot(kind="bar") 
        plt.savefig("day_0_unique.png", format='png')
        
        #plt.show()

def oneHotEncoding(clickDF  , columns):
    """
     Apply one-hot-code to given columns
    """
    
    allStages = [StringIndexer(inputCol=column, outputCol=column+STRING_INDEXER_OUT_SUFFIX).setHandleInvalid("skip") for column in columns]
    oneHotEncodeInputOutputNames = [(column+STRING_INDEXER_OUT_SUFFIX , column+ONE_HOT_ENCODER_OUT_SUFFIX) for column in columns]
    oneHotEncodeInputOutputNames = list(zip(*oneHotEncodeInputOutputNames))
    ohe = OneHotEncoderEstimator(inputCols=oneHotEncodeInputOutputNames[0] , outputCols=oneHotEncodeInputOutputNames[1])
    allStages.append(ohe);
    pipeline = Pipeline(stages=allStages)
    clickDF =  pipeline.fit(clickDF).transform(clickDF)
    deletedColumns = list(oneHotEncodeInputOutputNames[0])+columns; 
    return clickDF.drop(*deletedColumns)

def impute(clickDF  , numericColumnNames=[]):
    imputer =  Imputer().setInputCols(numericColumnNames).setOutputCols(numericColumnNames).setStrategy("mean")  
    return imputer.fit(clickDF).transform(clickDF)


def dataToVectorForLinear(clickDF , categoricalColumnsNames , numericColumnNames):
  """
   Data preprosesing phase. In there  , feature vectors is  created
  """    
  print (categoricalColumnsNames)
  clickDF = oneHotEncoding(clickDF,categoricalColumnsNames)
  clickDF = impute(clickDF,numericColumnNames)
  all_feature_columns = numericColumnNames + [columnName + ONE_HOT_ENCODER_OUT_SUFFIX for columnName in categoricalColumnsNames];
  feature_assembler = VectorAssembler(inputCols=all_feature_columns,outputCol="features")
  return feature_assembler.transform(clickDF);
  
def evaluateLogisticRegression(trainDF , testDF):
  """
   Traning by  logistic regression classifiers  with some regulization 
   params
   it returns bechmarkdata list for  regulization params
  """
  benchmarkData = []
  for reg in [00.1 ,1.0 ,10.0] :
    mlr = LogisticRegression( maxIter=10, regParam=reg, elasticNetParam=0.8, family="multinomial")
    model = mlr.fit(trainDF);
    predictions = model.transform(testDF).cache()
    print("Logistic Regrassion for reg {}".format(reg))
    accuracy = printevaluatation(model, predictions)
    benchmarkData += [("LR", "regulization" ,reg ,float(accuracy))]
  return benchmarkData 

def evaluateDecisionTree(trainDF ,testDF):
  """
   Traning by  decision tree classifiers  with some maxDepth 
   params
   it returns bechmarkdata list for  maxDepth params
  """
 
    benchmarkData = []
    for maxDepth in [10.0,15.0]:
      classifier = DecisionTreeClassifier(maxDepth=maxDepth)
      model = classifier.fit(trainDF)
      predictions = model.transform(testDF)
      print("Decision Tree evaluation with maxtDepth : {}".format(maxDepth))
      accuracy = printevaluatation(model, predictions)
      benchmarkData += [("DT", "maxDepth" ,maxDepth ,float(accuracy))]
    return benchmarkData
def evaluateRandomForest(trainDF ,testDF):
   """
   Traning by  random forest classifiers  with some numTree 
   params
   it returns bechmarkdata list for  numTree params
   """
 
    benchmarkData = []
    for numTree in [25.0]:
      classifier = RandomForestClassifier(numTrees = numTree);
      model = classifier.fit(trainDF)
      predictions = model.transform(testDF)
      print("Random Forest with numThree:{}".format(numTree))
      printevaluatation(model,predictions)
      accuracy = printevaluatation(model, predictions)
      benchmarkData += [("RDT", "numTree" ,numTree ,float(accuracy))]
    return benchmarkData
def evaluateGradientBoostTree(trainDF ,testDF):
    """
   Traning by  gradient boost tree classifiers  with some stepSize 
   params
   it returns bechmarkdata list for  stepSize params
   """
  
    benchmarkData = []
    for stepsize in [0.1]:
        classifier = GBTClassifier(stepSize=stepsize)
        model = classifier.fit(trainDF)
        predictions = model.transform(testDF)
        print("Gradient Boost Tree with stepsize : {}".format(stepsize))
        accuracy =  printevaluatation(model,predictions)
        benchmarkData += [("GBT", "sitepsize" ,stepsize ,float(accuracy))]
    return benchmarkData

def evaluateSVM(trainDF ,testDF):
    benchmarkData = []
    for reg in [0.1,1.0]:
       classifier = LinearSVC(regParam=reg)
       model = classifier.fit(trainDF)
       predictions = model.transform(testDF)
       print("SVM with reg: {}".format(reg))
       accuracy = printevaluatation(model, predictions)
       benchmarkData += [("SVM", "regulization" ,reg ,float(accuracy))]
    return benchmarkData;

def printevaluatation(model , predictions):
    """
      Print accuracy , precision , roc , recall from model.summary if it 
      exists. Also it calculate acuracy by using test data and return it
      
    """
    evalator_auc  = MulticlassClassificationEvaluator(labelCol="label" ,predictionCol="prediction")
    evalator_accuracy = MulticlassClassificationEvaluator(labelCol="label" ,predictionCol="prediction", metricName="accuracy")
    if(hasattr(model,"summary")):
        print("acurracy {}".format(model.summary.accuracy))
        print("roc {}".format(model.summary.roc))
        print("precision {}".format(model.summary.precisionByLabel))
        print("recall {}". format(model.summary.recallByLabel))
        print("under roc {}".format(model.summary.areaUnderROC))
    testError = 1- evalator_accuracy.evaluate(predictions)    
    print("test error {}".format(testError))
    # beacuse of spark performance and time limit, we comment code for uac. if you see it , please uncomment
    # print("test uac {}".format(evalator_auc.evaluate(predictions)))

    return testError;
    

     



print("============= READ Click ======");  
clickRDD = sc.textFile("s3a://wer-display-ads/day_0_1000000.csv"); 
#clickRDD = sc.textFile("data/day_0_1000.csv");
#clickRDD = sc.textFile("s3a://wer-display-ads/day_0.gz"); 

numericColumnNames , categoricalColumnsNames = getColumnStructure();
clickRDDRows = clickRDD.map(lambda line : lineToRow(line,labelColunmName="label" ,numericColumnNames=numericColumnNames , categoricalColumnNames=categoricalColumnsNames));
print("=============TO DF ======");
clickDF = sqlContext.createDataFrame(clickRDDRows)
clickDF = clickDF.cache()
""" some columns like have hig distinc values ( user id ) is filtered """
new_categoricalColumnsNames = [ x for x in categoricalColumnsNames if x not in ["c20", "c22", "c1" , "c10"]]

""" if you want to plot count of distinct values each column , please uncomment """""
#distinctCountDF = getColumnUniqueCounts(clickDF);
#distinctCountDF.write.csv("s3a://wer-display-ads/unique_result.csv",sep=",",header=True,mode="overwrite")
#distinctCountDF = distinctCountDF.toPandas()
#plotColumnUniqueCounts(distinctCountDF)

print("=============  TO FEATURE VECTOR  ======");
clickDF = dataToVectorForLinear(clickDF,new_categoricalColumnsNames,numericColumnNames);
trainDF , testDF = clickDF.randomSplit([0.8,0.2])
trainDF.cache()
testDF.cache()
bechmarkList = []

print("============= LOGISTIC REGRESSION ======");
benchmarkDataLR = evaluateLogisticRegression(trainDF, testDF)
bechmarkList += benchmarkDataLR
print("============= DECISION TREE ======");
benchmarkDataDT = evaluateDecisionTree(trainDF,testDF)
bechmarkList += benchmarkDataDT

print("=============RANDOM FOREST ======");
benchmarkDataRDT = evaluateRandomForest(trainDF,testDF)
bechmarkList += benchmarkDataRDT
print("============= GBT ======");
benchmarkDataGBT= evaluateGradientBoostTree(trainDF,testDF)
bechmarkList += benchmarkDataGBT
print("============= SVM ======");
benchmarkDataSVM = evaluateSVM(trainDF,testDF)
bechmarkList += benchmarkDataSVM

"""
Write benchmork result to s3
"""
bechmarkDF = sqlContext.createDataFrame(bechmarkList ,["model" ,"tunekey" ,"tunevalue" , "accuracy"])
bechmarkDF.write.csv(benchmarkOutputDirectory,sep=",",header=True,mode="overwrite")
