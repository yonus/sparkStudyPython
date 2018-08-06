import io, numpy, os;
from pandas import DataFrame;
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames , filenames in os.walk(path):
      for filename in filenames:
        path = os.path.join(root,filename)
        inBody = False;
        lines = [];
        f = io.open(path,"r", encoding="latin1")
        for line in f:
            if inBody:
                lines.append(line)
            elif line == "\n":
                inBody = True
        f.close()
        message = "\n".join(lines);
        yield  path , message

def dataFrameFromDirectory(path, classifier):
     rows = [];
     index = [];
     for filename , message in readFiles(path):
          rows.append({"message":message , "class":classifier})
          index.append(filename);
     return DataFrame(rows,index=index)
from numpy import random, array
def createClusterData(N , k):
   random.seed(10)
   pointCluster  = float(N)/k;
   X = []
   for i in range(k):
       incomeCentroid = random.uniform(20000.0 , 200000.0);
       ageCetroid = random.uniform(20.0, 70.0)
       for j in range (int(pointCluster)):
         X.append([random.normal(incomeCentroid,10000.0) , random.normal(ageCetroid,2.0)])
   X = array(X);
   return X;
print(createClusterData(100,5))   

data  = DataFrame({"message":[],"class":[]})
data = data.append(dataFrameFromDirectory("data/email/spam/" ,"spam"))
data = data.append(dataFrameFromDirectory("data/email/nspam/","nspam"))
data.head()


from  sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from sklearn.preprocessing  import scale

data = createClusterData()



