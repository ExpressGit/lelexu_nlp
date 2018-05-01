#coding:utf-8
import os
from matplotlib import pyplot
import scipy as sp
import numpy as np
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

''' 生成数据文件 
movie_reviews = load_files('tokens')
sp.save('movie_data.npy',movie_reviews.data)
sp.save('movie_target.npy',movie_reviews.target)
'''

# 读取数据
movie_data = sp.load('movie_data.npy')
movie_target = sp.load('movie_target.npy')
x = movie_data
y = movie_target

# 切分数据集，调用tfidfVector接口
count_vec = TfidfVectorizer(binary=False,decode_error='ignore',stop_words='english')

for i in range(10):
    #加载数据，切分数据集
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)

    #调用贝叶斯分类器
    clf = MultinomialNB()
    clf.fit(x_train,y_train)
    doc_class_predicted = clf.predict(x_test)

    print(np.mean(doc_class_predicted == y_test))

#准确率和召回率
precision,recall,thresholds = precision_recall_curve(y_test,clf.predict(x_test))
answer = clf.predict_proba(x_test)[:,1]
report = answer>0.5
print(classification_report(y_test,report,target_names=['neg','pos']))
