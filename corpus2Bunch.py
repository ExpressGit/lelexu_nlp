#coding:utf-8
import os
import sys
from sklearn.datasets.base import Bunch
import pickle as pickle

def _readfile(path):
    with open(path,"rb") as fp:
        content = fp.read()
    return content

def corpus2Bunch(wordbag_path,seg_path):
    catelist = os.listdir(seg_path)
    #new Bunch
    bunch = Bunch(target_name=[],label=[],filenames=[],contents=[])
    #extend fill list
    bunch.target_name.extend(catelist)

    #获取每个目录下的文件
    for mydir  in catelist:
        class_path = seg_path+mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path+file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))
    with open(wordbag_path,'wb') as file_obj:
        pickle.dump(bunch,file_obj)
    print("构建文本对象完毕")

if __name__ == "__main__":
    #train
    wordbag_path = "data/train_word_bag/train_set.dat"
    seg_path = "data/train_corpus_seg/"
    corpus2Bunch(wordbag_path,seg_path)

    # test
    wordbag_path = "data/test_word_bag/test_set.dat"
    seg_path = "data/test_corpus_seg/"
    corpus2Bunch(wordbag_path, seg_path)