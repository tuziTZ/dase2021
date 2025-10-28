
def a(n):
    if n==1:
        return 1
    else:
        return n*a(n-1)

print(a(5))
