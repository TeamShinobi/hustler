
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB 

import os
os.getcwd()

df = pd.read_csv("heart.csv") 
print(df)
print(df.count()) 
print(df.isna().sum())
print(df.describe())

##remove target column & set other columns or paramaeters as the training data
x = df.drop(['target'], axis=1) 
print(x)
## set target column as the training target/class
y = df[['target']] 
print(y)

## split dataset using hold-out set (80/20)
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.2) 

# #***************** train a DT model
dtc = DecisionTreeClassifier() 
dtc.fit(trainX, trainY) 

# ## predict the class of the hidden data i.e., 20% of the dataset
y_predicted = dtc.predict(testX)
print(y_predicted)

# ## test the classifier's perf via confusion matrix
[tp, fn, fp, tn] = confusion_matrix(testY,y_predicted, labels=[1,0]).ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
f1= 2*(precision*recall)/(precision+recall)

print ("precision : ", precision, "\n")
print ("recall : ", recall, "\n")
print ("specificity : ", specificity, "\n")
print ("harmonic mean : ", f1, "\n")


# #********************** train an NB model
nb = GaussianNB() 
nb.fit(trainX, trainY.values.ravel()) 

## predict the class of the hidden data i.e., 20% of the dataset
y_predicted = dtc.predict(testX)
print(y_predicted)

## test the classifier's perf via confusion matrix
[tp, fn, fp, tn] = confusion_matrix(testY,y_predicted, labels=[1,0]).ravel()


precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
f1= 2*(precision*recall)/(precision+recall)

print ("precision : ", precision, "\n")
print ("recall : ", recall, "\n")
print ("specificity : ", specificity, "\n")
print ("harmonic mean : ", f1, "\n")



# #********************** train an SVM model
svm = SVC(kernel='linear') 
svm.fit(trainX, trainY.values.ravel()) 

## predict the class of the hidden data i.e., 20% of the dataset
y_predicted = dtc.predict(testX)
print(y_predicted)

## test the classifier's perf via confusion matrix
[tp, fn, fp, tn] = confusion_matrix(testY,y_predicted, labels=[1,0]).ravel()


precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
f1= 2*(precision*recall)/(precision+recall)

print ("precision : ", precision, "\n")
print ("recall : ", recall, "\n")
print ("specificity : ", specificity, "\n")
print ("harmonic mean : ", f1, "\n")



# #********************** train a random forest model
rf = RandomForestClassifier() 
rf.fit(trainX, trainY.values.ravel()) 

## predict the class of the hidden data i.e., 20% of the dataset
y_predicted = dtc.predict(testX)
print(y_predicted)

## test the classifier's perf via confusion matrix
[tp, fn, fp, tn] = confusion_matrix(testY,y_predicted, labels=[1,0]).ravel()


precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
f1= 2*(precision*recall)/(precision+recall)

print ("precision : ", precision, "\n")
print ("recall : ", recall, "\n")
print ("specificity : ", specificity, "\n")
print ("harmonic mean : ", f1, "\n")

#********************** train a kNN model
knn = KNeighborsClassifier(n_neighbors=2) 
knn.fit(trainX, trainY.values.ravel()) 

## predict the class of the hidden data i.e., 20% of the dataset
y_predicted = knn.predict(testX)
print(y_predicted)

## test the classifier's perf via confusion matrix
[tp, fn, fp, tn] = confusion_matrix(testY,y_predicted, labels=[1,0]).ravel()


precision = tp/(tp+fp)
recall = tp/(tp+fn)
specificity = tn/(fp+tn)
f1= 2*(precision*recall)/(precision+recall)

print ("precision : ", precision, "\n")
print ("recall : ", recall, "\n")
print ("specificity : ", specificity, "\n")
print ("harmonic mean : ", f1, "\n")