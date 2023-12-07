# tuples can not be changed after it is created, often used when opbjects are made together
mytuple = ('mas',)
print(mytuple)

# making a list into a tuple, use tuple()
mylist = [1, 2, 3, 4]
newtuple = tuple(mylist)
print(newtuple)

# use indexing to get speific elements in the tuple like in list
item = newtuple[2]
print(item)

# tuples are iteranle
for i in newtuple:
    print(i)

# check if element is in in tuple with if when statement
if 1 in newtuple:
    print('yes')
else:
    print('no')

# Creating tuples with letters in it
lettertup = ('a', 'b', 'c', 'd', 'f', 'f')

# counting instances of element in tuple
print(lettertup.count('f'))

# finding index of an element in a tuple
# it is finding the first instance of 'b' and returning the index of 'b'
print(lettertup.index('b'))

# slicing tuples to get subparts using colons
# works like lists

supertup = (1, 2, 3, 45, 6, 7, 77)
b = supertup[1:5]
print(b)

# can use steps Start, End, Step
supertup = (1, 2, 3, 45, 6, 7, 77)
c = supertup[:6:2]
print(c)

# working with tuples can be more faster because it cant be changes. python will optimize it
