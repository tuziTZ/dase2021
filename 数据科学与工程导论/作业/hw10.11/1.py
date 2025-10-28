arr=[1,3,7,11,56,4,2,8,9,23,32,3,67,0,2,9]
def bsort(arr):
    for j in range(len(arr)-1):
        for i in range(len(arr)-1-j):
            if arr[i]>arr[i+1]:
                temp=arr[i]
                arr[i]=arr[i+1]
                arr[i+1]=temp
    return arr
print(bsort(arr))
