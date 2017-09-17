'''
Experiments with the numpy dot function.

numpy.dot(a, b, out=None)

Parameters:	
    a : array_like
    First argument.

    b : array_like
    Second argument.

    out : ndarray, optional
    Output argument. This must have the exact kind that would be returned if it was not used. In particular, it must have the right type, must be C-contiguous, and its dtype must be the dtype that would be returned for dot(a,b). This is a performance feature. Therefore, if these conditions are not met, an exception is raised, instead of attempting to be flexible.

Returns:	
    output : ndarray
    Returns the dot product of a and b. If a and b are both scalars or both 1-D arrays then a scalar 
    is returned; otherwise an array is returned. If out is given, then it is returned.

Raises:	
    ValueError
    If the last dimension of a is not the same size as the second-to-last dimension of b.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html

'''

import numpy as np

# Example with a 2 x 2 matrix.
#a = np.array([[1, 1],
#              [1, 1]], int)


#b = np.array([[2, 2],
#              [3, 3]], int)

# Example with a 2 x 1 matrix.
# Which will produce a scalar value (one dimensional). It will not be a vector or a matrix.
a = np.array([-1.02, -0.28], float)


b = np.array([5.1, 1.4], float)

print('The shape of a is: ' + str(a.shape))
print('The value of a is: ')
print(a)

print('The shape of b is: ' + str(b.shape))
print('The value of b is: ')
print(b)

dot_product = np.dot(b, a)

print(dot_product.shape)
print(dot_product)

