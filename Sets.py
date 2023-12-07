# sets are a datatype that is unordered and mutable, but cant make duplicates in values
myset = {1, 2, 3, 4, 5, 6, 7, 7, 7}
print(myset)

# can use the set function to make a set
myset2 = set([22, 33, 11, 22, 33, 44, 55, 22])
print(myset2)

# making emply set
myset3 = set()
myset3.add(22)
myset3.add(3)
print(myset3)

# to remove elements in a set
myset3.discard(22)
print(myset3)

# to clear set
myset3.clear()
print(myset3)

# iterate
for i in myset:
    print(i)

# if statement in set
if 22 in myset2:
    print('yes')
else:
    print('no')

# union and intersection
odd = {1, 3, 5, 7, 9}
evens = {2, 4, 6, 8, 10}
primes = {2, 3, 5, 7}

# unions combines elements from 2 sets, without duplication
u = odd.union(evens)
print(u)

# intersection only takes elements in both sets without duplications
x = odd.intersection(primes)
print(x)

# calculate the difference between 2 sets
d = odd.difference(primes)
print(d)

# symmetrix difference, returns elements from 2 sets, but not the elements in both set
# will not modify the original list

# to modify sets in place (union)
myset.update(myset2)
print(myset)

# interaction update modify sets in place (interaction)Updates sets with items in both sets
myset2.intersection(myset3)
print(myset2)

# using '.update', will modify the sets

# superset/ subset, return true of false. Determines if a sets values are all within another set.
# isdijoint, will let you know if elemesnts are the same as another set

# copying sets, the same as copying lists (.copy)
myset4 = myset2.copy()
print(myset4)
print(myset2)

# frozen set, immutable version of a normal set, cannot be changed
a = frozenset([1, 22, 44, 55, 1, 222, 3, 4, 2664, 22])
print(a)
