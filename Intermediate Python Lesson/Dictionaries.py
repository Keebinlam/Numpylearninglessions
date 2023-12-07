# a dictionary uses key value pairs, is unordered, amd changable
# key is mapped to a value
mydict = {'red': 'apple', 2: 'blue', 1: 'orange'}

# to get values of dictionayy, using keys
value = mydict[2]
print(value)

# adding new value to a dictionary
mydict['newkey'] = 'itsakey'
print(mydict)

# you can overwrite the value of a key
mydict['newkey'] = 'oldkey'
print(mydict)

# deleting key and values, use del
del mydict['newkey']
print(mydict)

# also use the .pop to delete item from the dictionary using a key
mydict.pop(1)
print(mydict)

# to delete the last item on the dictionary
mydict.popitem()
print(mydict)

mydict['newkey'] = 'oldkey'
mydict['yellow'] = 'print'
mydict['pizza'] = '44'
print(mydict)
# when checking if key are inside of dictionary
# if in statemet
# this, if true, will print the value associated with the key, if notm will print no
if 'name' in mydict:
    print(mydict['lastnanme'])
else:
    print('no')

# looping through a dictionary
for key in mydict:
    print(key)

# making a copy of a dictionary
mydictcopy = dict(mydict)
print(mydict)
mydictcopy.popitem()
print(mydictcopy)

# merging dictionary
# all the same key and value gets overwritten, and key/values that are no in the first list is added, adding the difference
mydict.update(mydictcopy)
print(mydict)

# keytypes
# numbers, tuples, strings, cant use list since it is not hashable
