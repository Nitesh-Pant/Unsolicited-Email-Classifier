import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV

dataframe=pd.read_csv("D:\College\Projects\spam.csv")
print(dataframe.head())

x=dataframe["EmailText"]
y=dataframe["Label"]

x_train, y_train= x[0:4457], y[0:4457]
x_test, y_test=x[4457:],y[4457:]

cv=CountVectorizer()
features=cv.fit_transform(x_train)

model=svm.SVC()
model.fit(features,y_train)

features_test=cv.transform(x_test)
print("Accuracy of the model" ,model.score(features_test,y_test))