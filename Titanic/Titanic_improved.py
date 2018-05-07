#coding:utf-8
import os
# 改进版
import numpy as np
import pandas as pd
from pandas import DataFrame
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model


# 重新做数据处理
# feature enginnering
#

# deal text
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return big_string
    return np.nan


enc = preprocessing.OneHotEncoder()


# print(df.info())
def clean_and_munge_data(df):
    # 处理缺省值
    df.Fare = df.Fare.map(lambda x: np.nan if x == 0 else x)
    # deal Name
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df['Title'] = df['Name'].map(lambda x: substrings_in_string(x, title_list))

    def replace_titles(x):
        title = x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df['Title'] = df.apply(replace_titles, axis=1)
    df['Family'] = df['SibSp'] * df['Parch']
    df['AgeFill'] = df['Age']
    mean_ages = np.zeros(4)
    #     print(df['Title'])
    mean_ages[0] = np.average(df[df['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df[df['Title'] == 'Mr']['Age'].dropna())
    mean_ages[2] = np.average(df[df['Title'] == 'Master']['Age'].dropna())
    mean_ages[3] = np.average(df[df['Title'] == 'Mrs']['Age'].dropna())
    df.loc[(df.Age.isnull()) & (df.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mr'), 'AgeFill'] = mean_ages[1]
    df.loc[(df.Age.isnull()) & (df.Title == 'Master'), 'AgeFill'] = mean_ages[2]
    df.loc[(df.Age.isnull()) & (df.Title == 'Mrs'), 'AgeFill'] = mean_ages[3]

    df['AgeCat'] = df['AgeFill']
    df.loc[(df.AgeFill <= 10), 'AgeCat'] = 'child'
    df.loc[(df.AgeFill > 60), 'AgeCat'] = 'aged'
    df.loc[(df.AgeFill > 10) & (df.AgeFill <= 30), 'AgeCat'] = 'adult'
    df.loc[(df.AgeFill > 30) & (df.AgeFill <= 60), 'AgeCat'] = 'senior'

    df.loc[(df.AgeFill <= 12), 'Child'] = 1
    df.loc[(df.AgeFill > 12), 'Child'] = 0
    df.loc[(df.Parch > 1) & (df['Title'] == 'Mrs'), 'Mother'] = 1
    df.loc[(df.Mother.isnull()), 'Mother'] = 0

    df.loc[(df.Fare.isnull()) & (df.Pclass == 1), 'Fare'] = np.median(df[df['Pclass'] == 1]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 2), 'Fare'] = np.median(df[df['Pclass'] == 2]['Fare'].dropna())
    df.loc[(df.Fare.isnull()) & (df.Pclass == 3), 'Fare'] = np.median(df[df['Pclass'] == 3]['Fare'].dropna())

    df.Embarked = df.Embarked.fillna('S')

    le = preprocessing.LabelEncoder()

    #     print(df.as_matrix(['Embarked'])[:,0].tolist())
    le.fit(df.as_matrix(['Embarked'])[:, 0].tolist())
    x_emb = le.transform(df.as_matrix(['Embarked'])[:, 0].tolist())
    df['Embarked'] = x_emb.astype(np.float)

    le.fit(df.as_matrix(['Sex'])[:, 0].tolist())
    x_sex = le.transform(df.as_matrix(['Sex'])[:, 0].tolist())
    df['Sex'] = x_sex.astype(np.float)

    le.fit(df.as_matrix(['Title'])[:, 0].tolist())
    x_title = le.transform(df.as_matrix(['Title'])[:, 0].tolist())
    df['Title'] = x_title.astype(np.float)

    le.fit(df.as_matrix(['AgeCat'])[:, 0].tolist())
    x_age = le.transform(df.as_matrix(['AgeCat'])[:, 0].tolist())
    df['AgeCat'] = x_age.astype(np.float)

    df = df.drop(['Name', 'Age', 'Cabin'], axis=1)  # remove Name,Age and PassengerId

    return df

#Bagging ,
#每次取训练集的一个subset，做训练，这样，我们虽然用的是同一个机器学习算法，但是得到的模型却是不一样的
df = pd.read_csv('data/Titanic/train.csv')
df = clean_and_munge_data(df)
train_df = df.filter(regex='Survived|AgeCat|SibSp|Parch|Fare|Embarked|Sex|Pclass|Mother|Family|Title')
# print(train_df.info())
train_np = train_df.as_matrix()
# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]
# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

df_test = pd.read_csv('data/Titanic/test.csv')
df_test = clean_and_munge_data(df_test)
test = df_test.filter(regex='AgeCat|SibSp|Parch|Fare|Embarked|Sex|Pclass|Mother|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':df_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("data/Titanic/logistic_regression_bagging_predictions.csv", index=False)