#coding:utf-8
import os

#选择排序
#时间复杂度 o(n^2)
#空间复杂度o(1)
def selectionSort(alist):
    for i in range(0,len(alist),1):
        min = alist[i]
        index = i
        print("i:"+str(i))
        for j in range(i,len(alist),1):
            if(alist[j]<min):
                min = alist[j]
                index = j
                print("index:"+str(index))
        tmp = alist[i]
        alist[i] = min
        alist[index] = tmp

#练习
def selectionSort2(alist):
    for i in range(0,len(alist),1):
        min = alist[i]
        index = i
        for j in range(i+1,len(alist),1):
            if(alist[j]<min):
                min = alist[j]
                index = j
        tmp = alist[i]
        alist[i] = min
        alist[index] = tmp

alist = [32,56,12,78.98,23,59]
selectionSort2(alist)
print(alist)