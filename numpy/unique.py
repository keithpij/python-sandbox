'''
Experiments with numpy.unique()

numpy.unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None)
Find the unique elements of an array.

Returns the sorted unique elements of an array. There are three optional outputs in addition to 
the unique elements: 
    The indices of the input array that give the unique values.
    The indices of the unique array that reconstruct the input array.
    The number of times each unique value comes up in the input array.

Parameters:	
    ar : array_like
    Input array. Unless axis is specified, this will be flattened if it is not already 1-D.

    return_index : bool, optional
    If True, also return the indices of ar (along the specified axis, if provided, or in the 
    flattened array) that result in the unique array.

    return_inverse : bool, optional
    If True, also return the indices of the unique array (for the specified axis, if provided) 
    that can be used to reconstruct ar.

    return_counts : bool, optional
    If True, also return the number of times each unique item appears in ar.

    axis : int or None, optional
    The axis to operate on. If None, ar will be flattened beforehand. Otherwise, duplicate items 
    will be removed along the provided axis, with all the other axes belonging to the each of the 
    unique elements. Object arrays or structured arrays that contain objects are not supported if 
    the axis kwarg is used.

Returns:	
    unique : ndarray
    The sorted unique values.
    
    unique_indices : ndarray, optional
    The indices of the first occurrences of the unique values in the original array. Only provided 
    if return_index is True.

    unique_inverse : ndarray, optional
    The indices to reconstruct the original array from the unique array. Only provided if 
    return_inverse is True.

    unique_counts : ndarray, optional
    The number of times each of the unique values comes up in the original array. Only provided 
    if return_counts is True.

'''

import numpy as np

print('Numpy version:  ' + np.version.version)

# Just passing in an array.
# both will produce [1 2 3] since the input array is flattened before it is evaluated.
u = np.unique([1, 1, 2, 2, 3, 3])
print(u)
print('\n')

a = np.array([[1, 1], [2, 3]])
u = np.unique(a)
print(u)
print('\n')

# The axis parameter tells unique to consider each array on the selected axis as opposed to
# each individual value in the ndarray.
# array([[1, 0, 0], [2, 3, 4]]) for axis = 0 
# using axis = 1 did not produce a result I could make sense of.
a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
u = np.unique(a, axis=0)
print(u)
print('\n')

# Returning the indecies of the of the original array that give the unique values. 
# Only the index of the first occurance will be returned.
a = np.array(['a', 'b', 'b', 'c', 'a'])
u, indices = np.unique(a, return_index=True)
print(u)
print(indices)
print(a[indices])
print('\n')

# Return the idecies of the unique array that reconstruct the original input value.
a = np.array(['a', 'b', 'b', 'c', 'a'])
print(a)
u, indices = np.unique(a, return_inverse=True)
print(u)
print(indices)
print(u[indices])
print('\n')

# Return the idecies of the unique array that reconstruct the original input value.
a = np.array(['a', 'b', 'b', 'c', 'a'])
print(a)
u, counts = np.unique(a, return_counts=True)
print(u)
print(counts)
print('\n')
