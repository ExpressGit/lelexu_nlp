#coding:utf-8
import os

import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

''' 数据读入 '''
data = []
labels = []
with open("1.txt") as files:
    for line in files:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)

''' 标签转换 0/1 '''
y[labels=='fat'] = 1

''''' 拆分训练数据与测试数据 '''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

''' 训练KNN 分类器 '''
clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
clf.fit(x_train,y_train)

''' 测试结果打印 '''
answer = clf.predict(x)
print(x)
print(answer)
print(len(answer))
print(y)
print(np.mean(answer==y))

''' 准确率和召回率 '''
# precision,recall,thresholds = precision_recall_curve(y_train,clf.predict(x_train))
# answer = clf.predict_proba(x)[:1]
# print(classification_report(y, answer, target_names = ['thin', 'fat']))
# print('precision:'+str(precision))

precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
answer = clf.predict(x)
print(classification_report(y, answer, target_names = ['thin', 'fat']))