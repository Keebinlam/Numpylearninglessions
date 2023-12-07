# https://www.youtube.com/watch?v=HGOBQPFzWKo
mylist = ['banana', 'cheery', 'apple']
print(mylist)

mylist2 = [5, True, 'apple']
print(mylist2)

item = mylist[0]
print(item)

for i in mylist:
    print(i)

    # to check if item is in list
if 'banana' in mylist:
    print('yes')
else:
    print('no')

# know the number of items in list
print(len(mylist))

# adding a new item to the list
mylist.append('lemon')

# inserting item in a location
mylist.insert(1, 'blueberry')
print(mylist)

# to remove last items
mylist.pop()

# to remove a specific item
mylist.remove('cherry')

# removing all elements
mylist.clear()

# reverse list
mylist.reverse()

# to sort list, to make list or in asending order, it will change orginal list
mylist.sort()

# to not make changes to list, but still want the ordered list
new_list = sorted(mylist)
print(new_list)

# to create new list with same elements
nylist = [0] * 5

# you can add list
superlist = mylist + mylist2

# slicing, to access subparts of list, spesificy start and end index, if not speification for start or end, it will either start at the begining or goes out to the end
a = mylist[:2]
print(a)
# you can do steps, (0:0:0) Start, End, Step

# making copy of list
listcopty = mylist.copy()
# can use slicing to get copy of list

# list comprihension, creating a new list using one line of code
secondlist = [1, 2,  3,  4, 5]
b = [number*number for number in secondlist]  # numbers are squared
# expression, then for in loop, then in list
