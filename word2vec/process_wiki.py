#coding:utf-8
import os

import logging
import os.path
import six
import sys

from gensim.corpora import WikiCorpus

#繁体转为简体字
#opencc -i wiki.zh.text -o wiki.zh.text.jian -c zht2zhs.ini

#分词 jieba
# 分词词典使用了130w+词典。分词代码：jieba.lcut(sentence)，默认使用了HMM识别新词；
# 剔除了所有非中文字符；
# 最终得到的词典大小为6115353；
# 目前只跑了64维的结果，后期更新128维词向量；
# 模型格式有两种bin和model；
# 下载链接：链接: https://pan.baidu.com/s/1o7MWrnc 密码:wzqv

#乱码问题
#iconv -c -t UTF-8 < wiki.zh.text.jian.seg > wiki.zh.text.jian.seg.utf-8
#http://www.52nlp.cn/中英文维基百科语料上的word2vec实验#more-8198

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) !=3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp,outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp,'w')
    # lemmatize 不做词干提取，提高速度
    wiki = WikiCorpus(inp,lemmatize=False,dictionary={})
    for text in wiki.get_texts():
        output.write(b' '.join(text).decode('utf-8') + '\n')
        i = i +1
        if(i % 10000 ==0):
            logging.info("Saved  "+str(i)+" articles")

    output.close()
    logger.info('Finished Saved '+str(i) + " articles")

# python process_wiki.py enwiki-latest-pages-articles.xml.bz2 wiki.en.text
