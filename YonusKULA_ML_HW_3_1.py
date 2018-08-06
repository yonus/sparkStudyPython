import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

housingDF = pd.read_csv("housing.csv");


#housingDF.hist(bins=50,figsize=(20,15))
housingDF.fillna(housingDF.mean(),inplace=True)

housingDF["income_cat"] = np.ceil(housingDF["median_income"] / 1.5)
#housingDF.where(housingDF["income_cat"] < 5 , 5.0,inplace=True)
#housingDF["income_cat"].hist()

Y = housingDF['median_house_value'];
house_features  = housingDF.drop(['median_house_value'],axis=1)
house_feature_categorial =  house_features["ocean_proximity"]
house_feature_numerical = house_features.drop(["ocean_proximity"],axis=1)
house_numerical_columns_list = list(house_feature_numerical);
print(house_numerical_columns_list)

standar_scaler = StandardScaler();
house_feature_numerical[house_numerical_columns_list] = standar_scaler.fit_transform(house_feature_numerical);

encoder = LabelBinarizer();
encoded_categorical_features = encoder.fit_transform(house_feature_categorial)
encoded_DF = pd.DataFrame(encoded_categorical_features, columns = ["ocean_proximity_"+str(int(i)) for i in range(encoded_categorical_features.shape[1])])

print(house_feature_numerical.shape)
print(encoded_categorical_features.shape)
final_house_features = pd.concat([house_feature_numerical,encoded_DF] ,axis=1)
print(final_house_features.shape)

X_train, X_test, y_train, y_test = train_test_split(final_house_features,Y,test_size=0.2);
estimator = sm.OLS(y_train
,X_train).fit()
print(estimator.summary())
#print(final_house_features)