'''
Dictionary Comprehensions
'''

# Constructing a dictionary using a comprehension.
d = { k:v for (k,v) in [(4,5), (1,2), (8,9)] }
print(d)

# Iterating over a dictionary keys in a comprehension.
d = { 2*x for x in {'a': 1, 'b':2, 'c':3}.keys()}
print(d)

# Iterating over a dictionary values in a comprehension.
d = { 2*x for x in {'a': 1, 'b':2, 'c':3}.values()}
print(d)

# Iterating over a dictionary items in a comprehension.
d = [ x for x in {'a': 1, 'b':2, 'c':3}.items()]
print(d)
