with open('en.txt','r',encoding='utf-8') as fp:
    s=fp.read()
s=s.replace('\n',' ')
print(s)
# with open('en.txt','r',encoding='utf-8') as fp:
#     fp.write(s)