#coding:utf-8
import os
import logging
import os
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ == "__main__":
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) <4:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp,outp1,outp2 = sys.argv[1:4]
    model = Word2Vec(LineSentence(inp),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())
    # Word2Vec(lines,sg=1,size=100,window=5,min_count=5,negative=3,sample=0.001,hs=1,workers=4)
    #sg=1 是skip-gram算法，默认sg=0,CBOW
    #size 输出词的向量维数，一般取100-200之间。太小，会导致词映射冲突，值太大会消耗内存
    #window 当前词和目标词之间的最大距离，3表示目标词前看3-b个词，向后看b个词。b [0,3]
    #min_count 是对词进行过滤，频率小于min_count的单词忽略
    #negative和sample 可根据训练结果进行微调，sample 表示更高频率词随机采样到设置的阀值
    #hs=1 softmax被采用，hs=0&negative!=0,则负采样将会被选择使用
    #workes 控制训练并行
    model.save(outp1)
    model.wv.save_word2vec_format(outp2,binary=False)
    model.vocabulary()

#python train_word2vec_model.py wiki.en.text wiki.en.text.model wiki.en.text.vector

