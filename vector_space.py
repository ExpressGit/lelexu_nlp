#coding:utf-8
import os

from sklearn.datasets.base import Bunch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def _readfile(path):
    with open(path,"rb") as fp:
        content = fp.read()
    return content

def _readbunchobj(path):
    with open(path,"rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

def _writebunchobj(path,bunchobj):
    with open(path,"wb") as file_obj:
        pickle.dump(bunchobj,file_obj)

def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):
    stpwrdlst = _readfile(stopword_path).splitlines()
    bunch = _readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,
                       tdf=[],vocabulary=[])

    #导入训练及的tf-idf的词向量空间
    if train_tfidf_path is not None:
        trainbunch = _readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_

    _writebunchobj(space_path,tfidfspace)
    print("tf-idf词向量空间创建OK")

if __name__ == "__main__":

    stopword_path = "data/train_word_bag/hlt_stop_words.txt"
    bunch_path = "data/train_word_bag/train_set.dat"
    space_path = "data/train_word_bag/tfidfspace.dat"
    vector_space(stopword_path,bunch_path,space_path)

    #test
    bunch_path = "data/test_word_bag/test_set.dat"
    space_path = "data/test_word_bag/testspace.dat"
    train_tfidf_path = "data/train_word_bag/tfidfspace.dat"
    vector_space(stopword_path,bunch_path,space_path,train_tfidf_path)

