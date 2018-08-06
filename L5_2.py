import numpy as np;
import pandas as pd;
from sklearn import tree;
from sklearn.preprocessing import LabelEncoder;
from IPython.display import Image
from sklearn.externals.six import StringIO;
from sklearn.ensemble import RandomForestClassifier
import  pydotplus
df = pd.read_csv("data/PastHires.csv" ,header=0);
labelEncoder = LabelEncoder();

df["Hired"] = labelEncoder.fit_transform(df["Hired"]);
df["Employed?"] = labelEncoder.fit_transform(df["Employed?"])
df["Interned"] = labelEncoder.fit_transform(df["Interned"])
df["Top-tier school"] =  labelEncoder.fit_transform(df["Top-tier school"])
df["Level of Education"] = labelEncoder.fit_transform(df["Level of Education"])

features = list(df.columns[:6])
Y  = df["Hired"]
X = df[features]
dtClassifier = tree.DecisionTreeClassifier()
dctModel = dtClassifier.fit(X,Y);

dot_data = StringIO();
tree.export_graphviz(dctModel,out_file=dot_data,feature_names=features)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img  = Image(graph.create_png())


randomForestClassifier = RandomForestClassifier(n_estimators=10)
rfModel =  randomForestClassifier.fit(X,Y)

#print(df.head());
