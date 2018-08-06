import numpy as np;
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns


def prepocessData(clickDataframe):
    return;

def getClickDataFrame(path , columns):
    if not path :
        raise Exception("Please enter file path")
    if not columns or len(columns) != 40 :
        raise Exception("Total column number should be 40")

    clickDataFrame = pd.read_csv(path,delimiter="\t" ,header=None , names=columns)
    return clickDataFrame       


def displayCategorialValueCount(uniqueValueSeries ,columns = None):
  """
   plot count of uniquevalues of categroical
   variables

   Parameters
   --------
   uniqueValueSeries : pandas.series 
        index of series indicate names of categrical 
   columns:list
       column names that is plotted
  """  
  ax = sns.barplot(uniqueCountDF.index, uniqueCountDF.values)
  positions  = range(len(uniqueCountDF.values))
  for tick,label in zip(positions,ax.get_xticklabels()):
    ax.text(positions[tick], uniqueCountDF.values[tick] + 0.05, uniqueCountDF.values[tick],
    horizontalalignment='center', size='small', color='b', weight='semibold')


def encodeWithDictVectorize(data , columns):
    """
     one-hot-code for string
    """
    labelTransformer = DictVectorizer()
    encodedValues = labelTransformer.fit_transform(data[columns].to_dict("records")).toarray()
    vec_data = pd.DataFrame(encodedValues , columns=labelTransformer.get_feature_names(), index=clickDataFrame.index);
    data.drop(columns ,axis=1,inplace=True)
    return pd.concat([data, vec_data], axis=1)


data_file_path = "data/day_0_100000.csv";
#data_file_path = "s3a://wer-display-ads/day_0_1000000.csv";

label_column = ["label"]
#Label	I1	I2	I3	I4	I5	I6	I7	I8	I9	I10	I11	I12	I13	C1	C2	C3	C4	C5	C6	C7	C8	C9	C10	C11	C12	C13	C14	C15	C16	C17	C18	C19	C20	C21	C22	C23	C24	C25	C26
numeric_columns = ["I"+ str(x) for x in range(1,14)]
categorical_columns = ["C"+str(x) for x in range(1,27)]
all_columns = label_column + numeric_columns + categorical_columns;

clickDataFrame = getClickDataFrame(data_file_path , all_columns)
clickDataFrame[numeric_columns].fillna(clickDataFrame[numeric_columns].mean(), inplace=True)
clickDataFrame[categorical_columns].fillna(clickDataFrame[categorical_columns].mode() ,inplace=True)
##clickDataFrame = clickDataFrame.dropna(how="any")
clickDataFrame[categorical_columns] = clickDataFrame[categorical_columns].astype('str').applymap(str.strip)
uniqueCountDF = clickDataFrame[all_columns].nunique();
print(clickDataFrame.head())
print(clickDataFrame.dtypes)
displayCategorialValueCount(uniqueCountDF)
encodedData = encodeWithDictVectorize(clickDataFrame,["C6","C9","C13", "C17","C19" ,"C26","C25"])
print(encodedData.shape)
plt.show()






