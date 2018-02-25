'''
Experiments with sets, lists, and tupbles.
'''

# Using the set constructor. Notice that the order of the values is in ascending order 
# when the set is printed out and that repeat values are removed.
x = set([6, 1, 2, 3, 4, 5, 4])
print(type(x))
print(x)

# The List constructor. Again, notice the order of the values. Order is not changed for lists. 
# Values can be repeated.
x = list((6, 1, 2, 3, 4, 5, 4))
print(type(x))
print(x)

# The tuple constructor. Order is not changed for tuples.
# Values can be repeated.
x = tuple([6, 1, 2, 3, 4, 5, 4])
print(type(x))
print(x)

