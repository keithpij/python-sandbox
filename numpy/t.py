'''
numpy.ndarray.T


The T property of an ndarray transposes a matrix or an array. Each value in a matrix is moved to a new
position in the matrix. The new position is calculated by interchanging the row and column indecies.

Note that T is a property that returns the transpose of an ndarray. It is not a function - no parenthasis are needed.
It does not act on the original value. The original value remains unchaged.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.T.html

'''

import numpy as np

x = np.array([[1,2],[3,4]])
print(x)

transpose_x = x.T
print(transpose_x)

y = np.array([1,2,3,4])
print(y)

transpose_y = y.T
print(transpose_y)
