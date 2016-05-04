# Built-in functions

# The print statement.
print('Hello World')

print(max('Keith'))

print(min('Keith'))

# Determining the type of a varialble.
# type() is a builtin function in Python.
print(type(25))

# Working with string variables.
s = 'Keith Pijanowski'
l = len(s)
print('The length of ', s, 'is ')
print(l)

# type conversion functions
s = '99'
i = int(s)
print(type(i))

i = 99
s = str(i)
print(type(s))

s = '3.14'
f = float(s)
print(type(f))

print(int(5.555))

# Random numbers
import random

print('Random Numbers')
print(random.random())

# Importing modules - Math functions


# Creating functions
def product(a, b):
    return a*b


def division(a, b):
    return a/b

# The function definition must occur before the call to the function.
v1 = 2
v2 = 3
p = product(v1, v2)

print 'The product of', v1, 'and', v2, 'is', p, '.'

# TODO Create more functions.
