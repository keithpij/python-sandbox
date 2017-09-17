'''
enumerating iterables using the built-in function enumerate.

enumerate(iterable, start=0)

Return an enumerate object. iterable must be a sequence, an iterator, or some other object 
which supports iteration. The __next__() method of the iterator returned by enumerate() 
returns a tuple containing a count (from start which defaults to 0) and the values obtained 
from iterating over iterable.

'''


# enumerate is equivalent to the following function.
def my_enumerate(sequence, start=0):
    n = start
    for elem in sequence:
        yield n, elem
        n += 1


seasons = ['Spring', 'Summer', 'Fall', 'Winter']

# Using the default value of start which is 0.
# [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
result = set(my_enumerate(seasons))
print(result)


# Note: start does not cause indecies bofore it to be skipped. Rather, it is the starting point
# for the first index.
# [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
result = list(enumerate(seasons, start=1))
print(result)
