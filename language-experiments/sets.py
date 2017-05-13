'''
Experiments with sets.
'''

s = {10,2,3,4,5}
print(s)

# Sets do not support indexing.
#print(s[0])

# Sets are mutable. add and remove do not take indecies. They take values which must exist
# in the set.
s.add(9)
s.remove(10)

# Sets can be summed if all members are numeric.
print(sum(s))

# You can test for membership.
print(5 in s)

# Determining the length of a set.
print(len(s))

a = {'a', 'b', 'c'}
b = {'c', 'd', 'e'}
print(a|b)

# Iterating over a set.
for x in s: 
    print(x)