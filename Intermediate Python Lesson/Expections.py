# a sytax error is when python detects an incorrect statement
# a = 5 print(1)
# print(a)) these are errors
# errors that occurs when the sytax is correct, but someelse trigger is called an expection error
# for example trying to add a number and a string raises a type error
# a = 5 +'10'
# importing a module that does not exist will give an error
# name error occurs when using a varible that is not deffined
# file not found error, an error can not be found in path
# value error: happens when argurment in function can not work
# a = [1,2,3]
# a.remove(4)
# print(a)
# index error: when trying to get the index of a value not in list
# to force an exception when an error has occured, use raise keyword

# x = -5
# if x < 0:
#     raise Exception('x should be positive')

# assert statement, but if it works, it wont show
# x = -5
# assert (x >= 0, 'fskfbksbfk')

try:
    a = 5 / 0
except:
    print('fsfksjf')

# The try statement in Python is used for exception handling, allowing developers to manage the handling of errors and exceptions that can occur during program execution. The main purpose of using try is to write code that is resilient to errors, meaning it can handle unexpected situations gracefully without crashing.
# can use else clause if there is not excetion clause
#can use finally cluase, will run even if there is no exception or not
