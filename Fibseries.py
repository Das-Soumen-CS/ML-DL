import sys

def Fib(n):
    if (n<0):
        print("Please provide the correct range")
    elif (n==0):
        n==0
        return 0
    elif (n==1) or (n==2):
        return 1
    else:
        
        return Fib(n-1)+Fib(n-2)
        
    pass

def main():
    n=sys.argv[1]
    print("This is my program name:",sys.argv[0])
    Val=Fib(int(n))
    print(Val)
    pass

main()