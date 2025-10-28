import sys
sys.setrecursionlimit(100000)
import time
arr=[]
N=10
for i in range(N):
    arr.append(i)
l=0
r=len(arr)-1
#顺序查找法
def search(arr,x):
    for i in arr:
        if i==x:
            return i
    return 0
#二分查找法
def binarySearch(arr, x, l, r):
    if r >= l:
        mid = int(l + (r - l) / 2)
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binarySearch(arr, x, l, mid - 1)
        else:
            return binarySearch(arr, x, mid + 1, r)
    else:
        return 0
t_a = time.perf_counter()
for i in range(N):
    search(arr,i)
t_b = time.perf_counter()
print("顺序查找所用时间为："+str((t_b - t_a)/N))

t_a = time.perf_counter()
for i in range(N):
    binarySearch(arr,i,l,r)
t_b = time.perf_counter()
print("二分查找所用时间为："+str((t_b - t_a)/N))