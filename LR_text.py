#coding:utf-8
import os
from numpy import *
from sklearn.datasets import load_iris

# load the dataset:iris
iris = load_iris()
samples = iris.data
print(samples)
target = iris.target
# print(target)

# 3 class

#import the logistic Regression
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression() # use LogisticRegression
classifier.fit(samples,target) #train model

x = classifier.predict([[5,3,5,2.5]])

print(x)

t = [i for i in range(0,5,1)]
print(t)
