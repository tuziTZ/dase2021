data=[2,2,2,2,2,2,1,2,2,2,2,2,2,2,2]
start=0
def b(data,start,end):
    if start>end:
        return -1
    mid=start+(end-start)//2
    if data[mid]==1:
        return mid
    elif sum(data[start:mid+1])<2*len(data[start:mid+1]):

        return b(data,start,mid)
    else:
        return b(data,mid+1,end)

print(b(data,0,len(data)-1))

