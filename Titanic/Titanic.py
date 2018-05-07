#coding:utf-8
import os
import pandas as pd #数据分析
import numpy as np # 科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt # 引入matplotlib包
plt.rcParams['font.sans-serif']=['SimHei']# 设置加载的字体名
plt.rcParams['axes.unicode_minus']=False #解决保存图像是负号'-'显示为方块的问题

# plt.rcParams['font.family']=['Times New Roman']
# print(font)
data_train = pd.read_csv("data/Titanic/Train.csv")
# 数据整体的情况
# print(data_train.info())
# 查看数值分布情况
# print(data_train.describe())

#图形化分析
fig = plt.figure()
fig.set(alpha=0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"获奖情况（1为获救）")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived,data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True,which='major',axis='y')
plt.title(u"按年龄看获救分布")

plt.subplot2grid((2,3),(1,0),colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde')
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客的年龄分布")
plt.legend((u"头等舱",u"2等舱",u"3等舱"),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各等船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

