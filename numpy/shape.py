'''
Experiments with numpy.ndarray.shape.

ndarray.shape

An ndarry property which can be set to a tuple indicating new array dimensions.

May be used to “reshape” the array, as long as this would not require a change in the 
total number of elements

Unlike "reshape" the shape property reshapes the current ndarray and does not return a new
array.
'''

import numpy as np

a = np.zeros((10, 2))
b = a.T
c = b.view()
c.shape = -1

print(a)
a.shape = 20
print(a)

# A transpose make the array non-contiguous
b = a.T
b.shape = (10,2)
print(b)

c = b.view()
c.shape = (20)
print(c)

# Taking a view makes it possible to modify the shape without modifying
# the initial object.
#c = b.view()


