import sklearn
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import tree
from sklearn import datasets
from sklearn.metrics import accuracy_score

url = "500_Person_Gender_Height_Weight_Index.csv"
df = pd.read_csv(url, sep = ",")
print(df.head(500))
label = df.iloc[:, 0:1]
label_train = label[0:400]
label_test = label[401:500]
print(label_test)
feature = df.drop("Gender",axis = 1)
feature_train = feature.loc[1:400,:]

# print("aaaaa")
# print(feature_train)
feature_test = feature.loc[401:500,:]
print(feature_test.head())
# print(df.head())

#linear regression model
# regr = linear_model.LinearRegression()
# regr.fit(feature_train, label_train)
# predict_regr = regr.predict(feature_test)

# decision tree model
cls = tree.DecisionTreeClassifier()
cls.fit(feature_train, label_train)
prediction_cls = cls.predict(feature_test)
accuracy_cls = accuracy_score(prediction_cls, label_test)
print(prediction_cls)
print(accuracy_cls)



#logistoic regression
lgr = linear_model.LogisticRegression()
lgr.fit(feature_train, label_train)
prediction_lgr = lgr.predict(feature_test)
print("logistic regression", prediction_lgr)
accuracy_lgr = accuracy_score(prediction_lgr, label_test)
print(accuracy_lgr)

# print(indicator.head())

# diabetes = datasets.load_diabetes()
# diabetes_X_train = diabetes.data[:-20]
# diabetes_X_test  = diabetes.data[-20:]
# diabetes_y_train = diabetes.target[:-20]
# diabetes_y_test  = diabetes.target[-20:]
# print(diabetes_y_train)