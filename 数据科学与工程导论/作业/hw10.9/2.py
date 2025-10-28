class Queue:
    def __init__(self):
        self.__items=[]
    def size(self):
        return len(self.__items)
    def empty(self):
        return len(self.__items)==0
    def In(self,x):
        self.__items.append(x)
    def Out(self):
        try:
            self.__items.reverse()
            self.__items.pop()
            self.__items.reverse()
        except:
            print("ERROR: Queue is empty now!")
    def Front(self):
        try:
            return self.__items[0]
        except:
            print("ERROR: Queue is empty now!")
    def Rear(self):
        try:
            return self.__items[-1]
        except:
            print("ERROR: Queue is empty now!")

q=Queue()
q.In(1)
q.In(2)
q.In(6)
print(q.size())
print(q.Front())
print(q.Rear())