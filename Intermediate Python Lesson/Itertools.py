# Iterators are objects that can be iterated upon, meaning that you can traverse through all the values
# They are designed to operate on elements of iterable objects, like lists or arrays, and they make Python code more efficient, readable, and concise.
# The itertools module is especially useful for working with large data sets efficiently, as many of its tools are implemented in C and optimized for performance. They are commonly used in data analysis, machine learning, and situations where complex iteration patterns are required
# iterators are dataypes that can be used in a for loop, like lists

from itertools import product
from itertools import permutations
from itertools import combinations
from itertools import combinations_with_replacement
from itertools import accumulate
from itertools import groupby

# product:Produces the Cartesian product of the provided iterables, equivalent to a nested for-loop.
a = [1, 2]
b = [3]
prod = product(a, b)
print(list(prod))
# can make it repeat, but becareful since the repeat can make the list large

prod2 = product(a, b, repeat=2)
print(list(prod2))

# permutations: return all possible ordering of an input
a = [1, 2, 3]
perm = permutations(a)
print(list(perm))

# can check for length of permutations
perm2 = permutations(a, 2)
print(list(perm2))

# combinations: Creates all possible combinations (without replacement) of r elements in the iterable.
# length is the second parameter, is required
c = [1, 2, 3, 4]
comb = combinations(c, 2)
print(list(comb))
# (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# to make the combinations repeat.
comb1 = combinations_with_replacement(c, 2)
print(list(comb1))
# this will make combinations with values of itself, like (1,1)

# makes an iterator that returns accumulated sums or any other binary function.
d = [1, 2, 3, 4]
acc = accumulate(d)
# this prints just the regular order of the list [1, 2, 3, 4]
print(d)
# this prints the sums of each [1, 3, 6, 10] (1+0=1),(1 + 2 = 3),(3 + 3 = 6), (6+4=10), and hypthetically if we had a 5 (10+5=15)
print(list(acc))

# groupby: makes an iteretor that returns keys and groups from iterable
# here we want to group a list of values based on the function
# if values from list g is smaller than 3, return true, then group by the function


def smaller_than_3(x):
    return x > 3


g = [1, 2, 3, 4]
grp = groupby(g, key=smaller_than_3)
for key, value in grp:
    print(key, list(value))

# False [1, 2, 3],True [4]

# they all feel like a type of for loop lol, but done so much easier
