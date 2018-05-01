#coding:utf-8
import os

###
#1、冒泡排序
#2、时间复杂度 O(n2)  空间复杂度O(1)
#冒泡排序的思想很简单，就是每次比较相邻两个元素的数据值将小的元素放置到数组的头部，将较大的元素放置到数组尾部。
# 因为这个过程小的元素一直从数据集里面上升到数据集头部很想气泡冒出水面因此称为冒泡排序。对于数量级为n的元素集合，
# 冒泡排序需要进行n-1次排列每一次都需要执行n此判断所以时间复杂度为0(n*n)，需要有一个临时空间做数据交换区因此空间复杂度为O(1)。
###
def bubbleSort(alist):
    for passnum in range(0,len(alist)-1,1):
        print("passnum:"+str(passnum))
        for i in range(0,len(alist)-passnum-1,1):
            print("i:"+str(i))
            if(alist[i] > alist[i+1]):
                tmp = alist[i+1]
                alist[i+1] = alist[i]
                alist[i] = tmp


# 练习
def bubbleSort2(alist):
    for i in range(0,len(alist)-1,1):
        for j in range(0,len(alist)-i-1,1):
            if(alist[j]>alist[j+1]):
                temp = alist[j]
                alist[j] = alist[j+1]
                alist[j+1] = temp

alist = [32,56,12,78.98,23,59]
bubbleSort2(alist)
print(alist)


