'''
Experiments with sets.
'''


s = {10, 2, 3, 4, 5}
print(s)

# Sets do not support indexing.
#print(s[0])

# Sets are mutable. add and remove do not take indecies. They take values which must exist
# in the set.
s.add(9)
s.remove(10)

# Sets can be summed if all members are numeric.
print(s)
print(sum(s))

# You can test for membership.
print(5 in s)

# Determining the length of a set.
print(len(s))

a = {0, 1, 2}
b = {2, 3, 4}
print('a = ' + str(a))
print('b = ' + str(b))

# Union operator
print('Union of a and b: ' + str(a | b))

# Intersection operator
print('Intersection of a and b: ' + str(a & b))

# The update method should be the same as union operator.
# a gets changed there is no return value.
a.update(b)
print('a.update(b): ' + str(a))

# The update_intersection method should be the same as the intersection operator.
# a gets changed there is no return value.
a = {0, 1, 2}
b = {2, 3, 4}
a.intersection_update(b)
print('a.intersection_update(b): ' + str(a))

# Iterating over a set.
for x in s: 
    print(x)

