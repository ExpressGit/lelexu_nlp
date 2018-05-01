#coding:utf-8
import os

#插入排序，假设A[0,1,....j-1] sorted sequece ,将A[j]插入进去
#
#
def insertionSort(alist):
    for i in range(1,len(alist),1):
        j = i-1
        key = alist[i]
        while (j>=0 and alist[j]>key):
            alist[j+1] = alist[j]
            j = j -1
        alist[j+1] = key

def insertionSort2(alist):
    for i in range(1,len(alist),1):
        j = i-1
        key = alist[i]
        while( j>=0 and alist[j]>key):
            alist[j+1] = alist[j]
            j = j -1
        alist[j+1] = key

def insertionSort3(alist):
    for i in range(1,len(alist),1):
        key = alist[i]
        j = i -1
        while(j>=0 and alist[j]>key):
            alist[j+1] = alist[j]
            j = j -1
        alist[j+1] = key

alist = [32,56,12,78,98,23,59]
insertionSort3(alist)
print(alist)