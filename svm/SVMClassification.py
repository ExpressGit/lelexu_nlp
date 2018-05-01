#coding:utf-8
import os
import numpy as np
import scipy as sp
from sklearn import svm
from sklearn.cross_validation import  train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def svm_prastice_one():
    data = []
    labels = []
    with open('1.txt') as file:
        for line in file:
            tokens = line.strip().split(' ')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)
    y[labels=='fat'] = 1
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    h=0.02
    #create a mesh to plot in
    x_min,x_max = x_train[:,0].min() - 0.1,x_train[:,0].max() + 0.1
    y_min,y_max = x_train[:,1].min() - 1,x_train[:,1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))

    ''' svm '''
    titles = ['LinearSVC (linear kernel',
              'SVC with polynomial (degree 3) kernel',
              'SVC with RBF kernel',
              'SVC with Sigmoid kernel'
              ]
    clf_linear = svm.SVC(kernel='linear').fit(x,y)
    clf_poly = svm.SVC(kernel='poly',degree=3).fit(x,y)
    clf_rbf = svm.SVC().fit(x,y)
    clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x,y)

    for i,clf in enumerate((clf_linear,clf_poly,clf_rbf,clf_sigmoid)):
        answer = clf.predict(np.c_[xx.ravel(),yy.ravel()])
        print(clf)
        print(np.mean(answer == y_train))
        print(answer)
        print(y_train)

        plt.subplot(2,2,i+1)
        plt.subplots_adjust(wspace=0.4,hspace=0.4)

        # put the result into a color plot
        z = answer.reshape(xx.shape)
        plt.contourf(xx,yy,z,cmap = plt.cm.Paired,alpha=0.8)

        plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap=plt.cm.Paired)
        plt.xlabel(u'身高')
        plt.ylabel(u'体重')
        plt.xlim(xx.min(),xx.max())
        plt.ylim(yy.min(),yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()

def svm_prastice_two():
    # 读取数据
    movie_data = sp.load('movie_data.npy')
    movie_target = sp.load('movie_target.npy')
    x = movie_data
    y = movie_target

    # 切分数据集，调用tfidfVector接口
    count_vec = TfidfVectorizer(binary=False, decode_error='ignore', stop_words='english')
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
    x_train = count_vec.fit_transform(x_train)
    x_test = count_vec.transform(x_test)

    ''' svm '''
    titles = ['LinearSVC (linear kernel',
              'SVC with polynomial (degree 3) kernel',
              'SVC with RBF kernel',
              'SVC with Sigmoid kernel'
              ]
    clf_linear = svm.SVC(kernel='linear').fit(x_train, y_train)
    clf_poly = svm.SVC(kernel='poly', degree=3).fit(x_train, y_train)
    clf_rbf = svm.SVC().fit(x_train, y_train)
    clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x_train, y_train)

    for i,clf in enumerate((clf_linear,clf_poly,clf_rbf,clf_sigmoid)):
        answer = clf.predict(x_test)
        print(titles[i])
        print(clf)
        print(np.mean((answer==y_test)))
        print(answer)
        print(y_test)


def svm_prastice_three():
    data = []
    labels = []
    h = 0.1
    ''' 数据生成 '''
    x_min,x_max = -1,1
    y_min,y_max = -1,1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                        np.arange(y_min,y_max,h))
    n = xx.shape[0]*xx.shape[1]
    x = np.array([xx.T.reshape(n).T,xx.reshape(n)]).T
    y = np.array([1 if item==False  else 0 for item in (x[:,0]*x[:,0]+x[:,1]*x[:,1]<0.8)])
    # y[y=="False"] = 0
    y.reshape(xx.shape)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    ''' svm '''
    titles = ['LinearSVC (linear kernel',
              'SVC with polynomial (degree 3) kernel',
              'SVC with RBF kernel',
              'SVC with Sigmoid kernel'
              ]
    clf_linear = svm.SVC(kernel='linear').fit(x, y)
    clf_poly = svm.SVC(kernel='poly', degree=3).fit(x, y)
    clf_rbf = svm.SVC().fit(x, y)
    clf_sigmoid = svm.SVC(kernel='sigmoid').fit(x, y)

    for i, clf in enumerate((clf_linear, clf_poly, clf_rbf, clf_sigmoid)):
        answer = clf.predict(x_test)
        print(clf)
        print(np.mean(answer == y_test))
        print(answer)
        print(y_test)

        # plt.subplot(2, 2, i + 1)
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)
        #
        # # put the result into a color plot
        # z = answer.reshape(x_test.shape)
        # plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
        #
        # plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)
        # plt.xlabel(u'身高')
        # plt.ylabel(u'体重')
        # plt.xlim(xx.min(), xx.max())
        # plt.ylim(yy.min(), yy.max())
        # plt.xticks(())
        # plt.yticks(())
        # plt.title(titles[i])

    plt.show()

if __name__ == "__main__":
    svm_prastice_three()