#coding:utf-8
import os
from multiprocessing import Pool,Queue,Process
import multiprocessing as mp
import time,random
import codecs
import jieba.analyse
import sys
jieba.analyse.set_stop_words('stop_word.txt')

def extract_keyword(input_string):
    tags = jieba.analyse.extract_tags(input_string,topK=100)
    return tags

def parallel_extract_keyword(input_string):
    tags = jieba.analyse.extract_tags(input_string, topK=100)
    return tags

#读取文件
def readfile(filepath):
    with open(filepath,encoding='gbk',errors="ignore") as fp:
        content = fp.read()
    return content

def savefile(savepath,content):
    with open(savepath,"a+") as fp:
        fp.write(content+"\n")



def genera_corpus(file_path_corpus,save_path,num=100):

    catelist = os.listdir(file_path_corpus)
    save_file = save_path
    #获取每个目录（类别）下的所有文件
    i = 0;
    for mydir in catelist:
        class_path = file_path_corpus + mydir +"/"

        if mydir ==".DS_Store":
            continue

        file_list = os.listdir(class_path)

        for file_path in file_list:
            fullname = class_path+file_path
            content = readfile(fullname)
            print(content)
            content = content.replace("\n","").replace("\r","").replace("\r\n","")
            content = content.replace(" ","")
            savefile(save_file,str(content))
            i = i+1
            if i == num:
                break;

if __name__ == "__main__":

    data_file = "message.txt"

    #生成语料文件
    corpus_path = "../data/train_corpus/"
    save_path = "message.txt"
    genera_corpus(corpus_path, save_path,100)


    with codecs.open(data_file) as file:
        lines = file.readlines()
        file.close()

    out_put = data_file.split('.')[0] + "_tags.txt"
    t0 = time.time()
    for line in lines:
        parallel_extract_keyword(line)
    print("串行处理花费时间{t}".format(t=time.time()-t0))

    #多线程处理
    pool = Pool(Process=int(mp.cpu_count()*0.7))
    t1 = time.time()
    res = pool.map(parallel_extract_keyword,lines)

    pool.close()
    pool.join()
    print("并行处理花费时间{t}".format(t=time.time() - t1))
