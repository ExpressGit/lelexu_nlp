#coding:utf-8
import os
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

''' 数据读入 '''
data = []
labels = []
with open ("1.txt") as file:
    for line in file:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])

x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)

'''' 标签转换0/1 '''
y[labels=='fat'] = 1
print(y)
''' 拆分训练数据 与 测试数据 '''
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

'''  使用信息墒作为划分的标准，对决策树进行训练 '''
clf = tree.DecisionTreeClassifier(criterion='entropy')
print(clf)
clf.fit(x_train,y_train)

''' 把决策树结构写入文件 '''
with open('tree.dot','w') as f:
    f = tree.export_graphviz(clf,out_file=f)

''' 系数反映每个特征的影响力，越大表示该特征在分类中起到的作用越大 '''
print(clf.feature_importances_)

''' 测试结果打印 '''
answer = clf.predict(x_test)
print(x_test)
print(answer)
print(y_test)
print(np.mean(answer == y_test))

''' 准确率与召回率计算 '''
precision,recall,thresholds  = precision_recall_curve(y_test,clf.predict(x_test))
answer = clf.predict_proba(x)[:,1]
print(classification_report(y, answer, target_names = ['thin', 'fat']))


