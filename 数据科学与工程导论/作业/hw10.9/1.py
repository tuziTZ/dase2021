str0='203.179.25.37'
slist=str0.split(".")
newlist=[]
for s in slist:
    a=int(s)
    keylist=[]
    for i in range(8):
        key=a%2
        a=a//2
        keylist.append(str(key))
    keylist.reverse()
    b="".join(keylist)
    newlist.append(b)
x='.'.join(newlist)
print(x)
