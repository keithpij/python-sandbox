'''
Experiments with zip.
'''

'''
A zip is a collection that can be iterated over.
A zip is created from multiple collections of the same length. 
Each element of the zip is a tuple which contains one element from the input collections.
'''
x = zip([1, 2, 3], ['A', 'B', 'C'])
print(x)
print(type(x))

for a, b in x:
    print(a, b)

# You have to unpack in the loop. The code below will not interate on x.
count = 0
for z in x:
    count += 1
    print(z)
print(count)