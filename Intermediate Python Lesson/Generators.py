# Generators in Python are a special type of function that allow you to declare a function that behaves like an iterator. They allow you to iterate through a set of values, but unlike lists, they don't store all values in memory at once. Instead, they generate each value on the fly and thus are more memory-efficient when dealing with large datasets
# use cases for generators
# 1) processing large datasets
# 2) memeory pipeline
# 3) Infinite sequence
# 4) perforamce optimazation
# used "yeild" keyword instead of "return"

def mygen():
    yield 1
    yield 2
    yield 3


g = mygen()
print(g)

for i in g:
    print(i)

# you can have the function run one value at a time by calling the for it to run next

d = mygen()
value = next(d)
print(value)

value = next(d)
print(value)

value = next(d)
print(value)

# if I run it passed the total yield count, the function will iniatate the stop iteration

# can use as inputs in iterable functions

p = mygen()
print(sum(p))

# Remembering the current state


def countdown(num):
    print('starting')
    while num > 0:
        yield num
        num -= 1


cd = countdown(4)
ap = next(cd)
print(ap)

print(next(cd))


# big advantage of generators are they are very memory effieenet, save alot of memeory data
# this uses a lot of memory for storing the list
def firstn(n):
    nums = []
    num = 0
    while num < n:
        nums.append(num)
        num += 1
    return nums


print(sum(firstn(10)))

# this is the same as the last code, expect more memeory effienct, because its not saving the list in an array


def firstn_gen(n):
    num = 0
    while num < n:
        yield num
        num += 1


print(sum(firstn_gen(10)))

# fibonacci sequence in generator


def fibonacci(limit):
    # first number is 0 and 1, sum of previence number
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


fib = fibonacci(30)
for i in fib:
    print(i)


# Generator expressions
# taking all the even values from 1 = 10
mygena = (i for i in range(10) if i % 2 == 0)
for i in mygena:
    print(i)

# parameters
# the 'name' in the parathesis would be the parameter


def printname(name):
    print(name)


# arguements
# the 'kevin' in parathesis is the arguement, which is called in a function. "put kevin in the functions"
printname('kevin')

# kwargs

# having default value for d, will print d no matter what, but is not requried in argurements


def foo(a, b, c, d=99):
    print(a, b, c, d)


# here a 'key" is assigned to and linked to a parameter.
foo(c=11, a=12, b=3)

# in a function, if you add *args, or **kwargs, you can pass any number of argyemenbt or kwward arguments in the function


def wosmd(a, b, *args, **kwargs):
    print(a, b)  # print the parameters a and B
    for arg in args:
        print(args)  # will print as tuple
    for key in kwargs:
        print(key, kwargs[key])  # print as dictionary


#it first prints the first 2 positiona agruements, then prints the tuple argument, then lastly the kwarg
wosmd(1, 2, 11, 22, 33, three=2, seven=32)

