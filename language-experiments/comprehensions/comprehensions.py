'''
Experiments with list comprehensions.
'''

result = [2*x for x in {2, 1, 3, 4, 5}]
print('List comprehension: ' + str(result))

# Note that since it is a list that is iterated over the order is preserved.
result = [2*x for x in [2, 1, 3, 4, 5]]
print('Set comprehension: ' + str(result))

result = [x*y for x in [1, 2, 3] for y in [10, 20, 30]]
print(result)

# Note that since the result is a set there are no duplicate values in the result.
result = {x*y for x in {1,2,3} for y in {10,20,30}}
print(result)

result = [[x, y] for x in ['A', 'B', 'C'] for y in [1, 2, 3]]
print(result)

# Use a comprehension to sum all the numbers in a list of lists.
LofL = [[.25, .75, .1], [-1, 0], [4, 4, 4, 4]]
result = sum([sum(l) for l in LofL])
print(result)

# Using a list for unpacking.
[x, y, z] = [4*1, 4*2, 4*3]
print(x)

# Using a list comprehension for unpacking.
list_of_lists = [[1, 1], [2, 4], [3, 9]]
result = [y for [x,y] in list_of_lists]
print(result)

# This produces a generator.
result = (i for i in [1, 2, 3])
print(type(result))