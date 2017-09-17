'''
Experiments with Numpy.ravel().

numpy.ravel(a, order='C')

Returns a one dimensional array of type numpy.ndarray.

Parameters:	
a : array-like
In Python an array-like object is any Python object that np.array can convert to an 
ndarray. The elements in a are read in the order specified by order, and packed as a 1-D array.

order : {‘C’,’F’, ‘A’, ‘K’}, optional
The elements of a are read using this index order. ‘C’ means to index the elements in row-major, 
C-style order, with the last axis index changing fastest, back to the first axis index changing slowest. 
‘F’ means to index the elements in column-major, Fortran-style order, with the first index changing 
fastest, and the last index changing slowest. Note that the ‘C’ and ‘F’ options take no account 
of the memory layout of the underlying array, and only refer to the order of axis indexing. 
‘A’ means to read the elements in Fortran-like index order if a is Fortran contiguous in memory, 
C-like order otherwise. ‘K’ means to read the elements in the order they occur in memory, except 
for reversing the data when strides are negative. By default, ‘C’ index order is used.

Returns:	
y : numpy.ndarray
Regardless of whether a is a matrix or a 1-D array, the return value y is a 1-D array of type numpy.ndarray,
The shape of the returned array is (a.size,). Matrices are special cased for backward compatibility.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html#numpy.ravel

'''

import numpy as np

x = np.array([[1, 3, 2], [7, 5, 6.1]])
y = np.ravel(x)
print(y)
print(type(x))
print(x.shape)

# Create a 2 x 2 matrix.
simple_array = np.array([1, 2], float)
y = simple_array.ravel()
print(y)
print(type(y))
print(y.shape)

