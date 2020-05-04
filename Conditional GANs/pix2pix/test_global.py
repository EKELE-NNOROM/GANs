
def multiply(a, b):
    global c
    c = 100
    return (a*b)

def division(d):
    return (c/d)




if __name__ == "__main__":
    print(multiply(5, 10))
    print(division(5))
    print(c)
