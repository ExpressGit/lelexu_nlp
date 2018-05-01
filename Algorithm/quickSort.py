#coding:utf-8
import os
#1、先从数列中取出一个数作为基准数
#2、分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边
#3、再对左右区间重复第二步，直到各区间只有一个数
#https://blog.csdn.net/taotaoah/article/details/50987837
#1．i =L; j = R; 将基准数挖出形成第一个坑a[i]。
#2．j--由后向前找比它小的数，找到后挖出此数填前一个坑a[i]中。
#3．i++由前向后找比它大的数，找到后也挖出此数填到前一个坑a[j]中。
#4．再重复执行2，3二步，直到i==j，将基准数填入a[i]中。
#时间复杂度 nlogn 空间复杂度 o(n)

def partitions(alist,p,r):
    print('start:' + str(p))
    print('end:' + str(r))

    i = p-1
    pivot = alist[r]  #基准数
    for j in range(p,r,1):
        if (alist[j] < pivot):
            i = i+1
            tmp = alist[j]
            alist[j] = alist[i]
            alist[i] = tmp
    tmp = alist[r]
    alist[r] = alist[i+1]
    alist[i+1] = tmp
    print("q:"+str(i+1))
    return i+1

#练习
def partitions2(alist,p,r):
    i = p -1
    pivot = alist[r] #基准数
    for j in range(p,r,1):
        #注意i的指针，只有位置移动时，i才会自增
        #所以循环结束后，i+1 就是分界位置
        if(alist[j] < pivot):
            i = i+1
            tmp = alist[j]
            alist[j] = alist[i]
            alist[i] = tmp
    #基准数处理，插入到数组的分界线中，左侧小于他，右侧大于他
    tmp = alist[r]
    alist[r] = alist[i+1]
    alist[i+1] = tmp
    return i+1



def quickSort(alist,p,r):
    if(p<r):
        q = partitions2(alist,p,r)
        quickSort(alist,p,q-1)
        quickSort(alist,q+1,r)



alist = [32,56,12,78,98,23,59]
quickSort(alist,0,6)
print(alist)