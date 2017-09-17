'''
Experiments with numpy.where.

numpy.where(condition[, x, y])

Return elements, either from x or y, depending on condition.
If only condition is given, return condition.nonzero().

Parameters:	

    condition : array_like, bool
    When True, yield x, otherwise yield y.

    x, y : array_like, optional
    Values from which to choose. x, y and condition need to be broadcastable to some shape.

Returns:	

    out : ndarray or tuple of ndarrays
    If both x and y are specified, the output array contains elements of x where condition is True, 
    and elements from y elsewhere.
    
    If only condition is given, return the tuple condition.nonzero(), the indices where 
    condition is True.

'''

import numpy as np

x = np.arange(9.).reshape(3, 3)
print(x)

# Returns a tuple of arrays that contains the row and column locations in x where x > 5.
#print(np.where( x > 5 ))

# Produces a 1D array of all the values in x that are greater than 3.
# The use of the np.where is used to produce the indecies into x. It returns a tuple of indecies
# which when used as the index values for x (an ndarray) will result in a single dimension
# array.
print(x[np.where( x > 2.0 )])             

# 
#print(np.where(x < 5, x, -1))               # Note: broadcasting.
