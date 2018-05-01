#coding:utf-8
import os
from math import floor

#基于一个有序序列进行查找，针对一个有序序列
#二分查找法
def binarySearch(alist,n,low,high):
    if(low<high):
        mid = floor((high+low)/2)
        if(alist[mid]>n):
            binarySearch(alist,n,low,mid)
        else:
            binarySearch(alist,n,mid+1,high)
        if(alist[mid] == n):
            print("数组第："+str(mid))
            return mid

def binarySearch2(alist,num,low,high):
    if(low<high):
        mid = floor((low+high)/2)
        if(alist[mid]>num):
            binarySearch2(alist,num,low,mid)
        else:
            binarySearch2(alist,num,mid+1,high)
        if(alist[mid] == num):
            print("第："+str(mid))
            return alist[mid]

alist = [1,2,3,4,5,6,7,8,9]
binarySearch2(alist,7,0,8)
