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

array = df.values

#split the data into test and train
feature = array[:, 1:4]
print(feature)
label = array[:, 0:1]
print(label)
validation_size = 0.2
feature_train, feature_test, label_train, label_test = model_selection.train_test_split(feature, label, test_size = validation_size, random_state = 42)


#brutal force data split

# label = df.iloc[:, 0:1]
# label_train = label[0:400]
# label_test = label[401:500]
# print(label_test)
# feature = df.drop("Gender",axis = 1)
# feature_train = feature.loc[1:400,:]
# feature_test = feature.loc[401:500,:]
# print(feature_test.head())
# print(df.head())

#linear regression model  does not apply here unless i transform the label into float
# regr = linear_model.LinearRegression()
# regr.fit(feature_train, label_train)
# predict_regr = regr.predict(feature_test)

# decision tree model
cls = tree.DecisionTreeClassifier()
cls.fit(feature_train, label_train)
prediction_cls = cls.predict(feature_test)
accuracy_cls = accuracy_score(prediction_cls, label_test)
cv_cls = model_selection.cross_val_score(cls, feature_train, label_train, cv = 10) #cv should be within the range of number of samples
print("prediction for decision tree is:", prediction_cls)
print("accuracy for decision tree is:", accuracy_cls)
print("cross validation for decision tree is:", cv_cls)



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