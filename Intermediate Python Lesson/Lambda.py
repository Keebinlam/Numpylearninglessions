# lambda function is a small anonymous function, which means it doesn't require a formal function definition with a def keyword and doesn't need a specific name. Lambda functions can have any number of arguments but only one expression.
# lambda arguments: expression (format)
# Lambda functions are most useful when you need a small function for a short duration and you want to avoid the verbosity of regular function syntax. However, they are limited in their capabilities and aren't suitable for complex functions, as they can make the code less readable.
from functools import reduce
def add10(x): return x + 10


print(add10(5))

# the same as


def add_10func(x):
    return x + 10


print(add_10func(11))

# can have multiple arguments


def mult(x, y): return x*y


print(mult(2, 7))

# it is good for cases when the function is used once in the code

# sorted with lamda, sorting the values by the Y index instead
a = [(1, 200), (11, 7), (45, 13), (5, -1), (60, 1), (-3, 2)]
asort = sorted(a, key=lambda x: x[1])
print(a)
print(asort)

# same as this function


def sortbyy(x):
    return x[1]


bsort = sorted(a, key=sortbyy)
print(bsort)

# sorting with sum of list
csort = sorted(a, key=lambda x: x[0] + x[1])
print(csort)

dsort = sorted(a, key=lambda x: sum(x))
print(dsort)

# map: transform each element with a function
# map(func, seq)
# have to conver to a list to see the results
we = [1, 2, 3, 4, 5]
ew = map(lambda x: x * 2, we)
print(list(ew))

# same things as the one above, exepct its called'list comperhension"
# for loop
wr = [x * 2 for x in we]
print(wr)

# filter (func, seq)
# returns all elements where the element is true
qw = [1, 2, 3, 4, 5]
wq = filter(lambda x: x > 3, qw)
print(list(wq))

# same thing as before but with for loop
rt = [x for x in qw if x > 3]
print(rt)

# reduce (func, seq)
# repeatedly applies the function to the element and returns a single value
xc = [1, 2, 3, 4, 5]
producta = reduce(lambda x, y: x * y, xc)
print(producta)
