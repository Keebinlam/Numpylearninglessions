# #A decorator is a function that takes another function as an argument and extends the behavior of this latter function without explicitly modifying it.
# How Decorators Work
# Definition: A decorator is defined as a function.
# Application: It is applied to another function using the @decorator_name syntax just above the function definition.
# Execution: When the decorated function is called, it's passed through the decorator before executing.

# There are two commons decorations, function decorator and class

# function decortator
# a decorator takes a function, and another function, as an arguement, and extends its ability

import functools


def startswith(func):

    def wrapper():
        print('Start')
        func()
        print('end')
    return wrapper


def printname():
    print('kevin')


printname = startswith(printname)


# the code on top is the same at this code
def startswith(func):

    def wrapper():
        print('Start')
        func()
        print('end')
    return wrapper


@startswith
def printname():
    print('kevin')


printname()

# templete for a decorator


def my_decorator(func):

    @functools.wraps(func)
    def wrapper(*arges, **kwargs):
        # do something before
        result = func(*arges, **kwargs)
        # do something after
        return result
    return wrapper
