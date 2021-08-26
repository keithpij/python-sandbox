'''
Experiments with reversed.
'''

# Iterates of a list in reverse order.
y = [x*x for x in reversed([4, 5, 10])]
print(y)

# reversed can be used to iterate over a tuple in reverse order.
y = [x*x for x in reversed((4, 5, 10))]
print(y)

d = dict()
d[4] = '4'
d[0] = '0'
d[1] = '1'
d[2] = '2'
print(d)

# A dictionary is not reversable. The code below will throw an error.
#y = [x*x for x in reversed(d)]
#print(y)
