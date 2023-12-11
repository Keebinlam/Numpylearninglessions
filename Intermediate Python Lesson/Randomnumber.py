# random module is a library to generate random numbers
import random
import secrets
import numpy

# psuedo random numbers seem random but are preproducable

a = random.random()
# this will print a float from 0-1
print(a)

b = random.randint(1, 11)
# this will produce a number within the range 1- 11
# includes the upper range
print(b)

c = random.normalvariate(0, 1)
# useful in working in stat, allowing to add mew and sigma
# a mew of 0 and a standard divation of 1
print(c)


# from a list of elements, random.choice allows you to put in the list that rand will pick from
mylist = list("abdkdsfhljhfs")
print(mylist)

d = random.choice(mylist)
print(d)

# for the list of elements, pick 4 elements at random
f = random.sample(mylist, 4)
print(f)

# random shuffle, it will shuffle a list in place
mylist1 = list("abdkdsfhljhfs")
random.shuffle(mylist1)
print(mylist1)

# random seed will let you reproduce the numbers generated
random.seed(1)
k = random.randint(1, 110)
print(k)

# sercets module is used for PW, authenication, generate a true random numbner
l = secrets.randbelow(10)
print(l)

# you can use numpy to work with arrays with random floats
ww = numpy.random.randint(1, 10, (8, 5))
print(ww)

# shuffling an arrays

qw = numpy.array([[1, 2], [3, 10], [21, 4], [52, 11]])
numpy.random.shuffle(qw)
print(qw)
