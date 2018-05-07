#coding:utf-8
import os
import sys
import jieba

def savefile(savepath,content):
    with open(savepath,"wb") as fp:
        fp.write(content.encode('utf-8'))


#读取文件
def readfile(filepath):
    with open(filepath,encoding='gbk',errors="ignore") as fp:
        content = fp.read()
    return content

#语料分词
def corpus_segment(corpus_path,seg_path):
    '''

    :param corpus_path: 未分词的语料库路径
    :param seg_path: 分词后语料库存储路径
    :return:
    '''
    catelist = os.listdir(corpus_path)

    #获取每个目录（类别）下的所有文件
    for mydir in catelist:
        class_path = corpus_path + mydir +"/"
        seg_dir = seg_path+mydir+"/"

        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if mydir ==".DS_Store":
            continue

        file_list = os.listdir(class_path)

        for file_path in file_list:
            fullname = class_path+file_path
            content = readfile(fullname)
            print(content)
            content = content.replace("\r\n","")
            content = content.replace(" ","")
            content_seg = jieba.cut(content)
            savefile(seg_dir+file_path," ".join(content_seg))

    print('语料处理完成')

if __name__ == "__main__":
    #训练集
    corpus_path = "./data/train_corpus/"
    seg_path = "./data/train_corpus_seg/"
    corpus_segment(corpus_path,seg_path)

    #测试集
    corpus_path = "./data/test_corpus/"
    seg_path = "./data/test_corpus_seg/"
    corpus_segment(corpus_path,seg_path)