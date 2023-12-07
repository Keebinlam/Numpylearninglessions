# a string is an ordered, immutable collection datatype that is used for text representation
mystring = 'Hello World'

# tripple quotes """"fdsds""""
# \ is used to keep the string in one line

# accessing parts of strings like in list
chara = mystring[6]
print(chara)

# slicing strings
substring = mystring[:5]
print(substring)

# can use steps
stepstring = mystring[::2]
print(stepstring)

# concatinating strings
name = 'tom'
greeting = 'Hello'
sentence = greeting + ' ' + name
print(sentence)

# for loop strings
for i in sentence:
    print(i)

# check if a letter or text is in string
if 't' in sentence:
    print('yes')
else:
    print('no')

# putting spaces in the string, will count as part of the string
mystring2 = '        dsadadasd        '
print(mystring2)

# to remove the spaces, use .strip, this is just setting a new string with the removed values
mystring2 = mystring2.strip()
print(mystring2)

# converting string to upper case
mystringupper = mystring.upper()
print(mystringupper)

# converting string to lower case
mystringlower = mystringupper.lower()
print(mystringlower)

# to capitize the first letter
mystringcap = mystringlower.capitalize()
print(mystringcap)

# checking if strong starts with (true or false)
print(mystringcap.startswith('H'))


# checking if strong ends with (true or false)
print(mystringcap.endswith('worls'))

# finding the index in a string, should the index of the first instance
print(mystringcap.find('e'))

# counting number of strings
print(mystringcap.count('o'))

# lists and strings
# turning elements of a string into a list
mysentence = ('hello how are you doing?')
mylist = mysentence.split()
print(mylist)
# the delimiter is a space, and splits strings into a list by space

# convering a list into a new string
a = ('how,are,you,doing')
b = a.split(',')
print(b)
# adding spaces between each element so it goes back into a string
c = ' '.join(b)
print(c)

# formatting a string using f string
aname = 'tome'
sentence2 = f'hello, {aname}, how are you?'
print(sentence2)
