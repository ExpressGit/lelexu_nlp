#coding:utf-8
import os
import jieba
import jieba.posseg as pseg
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


if __name__ == "__main__":
    corpus = ["我 来到 北京 清华大学",
              "他 来到 来 网易 杭研 大厦",
              "小明 硕士 毕业 与 中国 科学院",
              "我 爱 北京 天安门"
              ]
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    # get tfidf marix
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    word = vectorizer.get_feature_names()
    print(word)
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的词频
    weight = tfidf.toarray()
    print(weight)

    for i in range(len(weight)):
        print(u"-----------这里输出第",i,u"类文本的词语权重------")
        for j in range(len(word)):
            print(word[j],weight[i][j])