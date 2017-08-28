'''
numpy.reshape

numpy.reshape(a, newshape, order='C')

Gives a new shape to an array without changing its data.

Parameters:	
    a : array_like
    Array to be reshaped.

    newshape : int or tuple of ints
    The new shape should be compatible with the original shape. If an integer, then the result 
    will be a 1-D array of that length. 
    One shape dimension can be -1. In this case, the value is inferred from the length 
    of the array and remaining dimensions.
    
    order : {‘C’, ‘F’, ‘A’}, optional
    Read the elements of the 'a' parameter using this index order, and place the elements 
    into the reshaped array using this index order. ‘C’ means to read / write the elements 
    using C-like index order, with the last axis index changing fastest, back to the first 
    axis index changing slowest. ‘F’ means to read / write the elements using Fortran-like 
    index order, with the first index changing fastest, and the last index changing slowest. 
    Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying 
    array, and only refer to the order of indexing. ‘A’ means to read / write the elements in 
    Fortran-like index order if a is Fortran contiguous in memory, C-like order otherwise.

Returns:	
    reshaped_array : ndarray
    This will be a new view object if possible; otherwise, it will be a copy. Note there is 
    no guarantee of the memory layout (C- or Fortran- contiguous) of the returned array.


https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html

'''

import numpy as np

a = np.zeros((10, 2))
print(a)
a.shape = 20
print(a)

# A transpose make the array non-contiguous
b = a.T
print(b)

# Taking a view makes it possible to modify the shape without modifying
# the initial object.
#c = b.view()

c = np.reshape(c,(5,4))
print(d)

#c.shape = (20)
