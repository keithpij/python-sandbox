'''
arange Experiments

arange takes up to four parameters: start, stop, step, dtype and returns an numpy.ndarray of evenly
spaced values. The values generated will be what is known as a half open stop. In other words the 
array will include the start value but not the stop value.

start is the start value for the sequence of numbers that will be in the return value. This value will be
the first value in the array. (The 0 index.) If it is not specified then 0 is the default value.

stop is the stop value for the sequence of numbers generated and returned by arange. 
It must be specified. This value will not be in the returned ndarray.

step is the spacing between generated values. If it is not specified then the default value will
be used which is 1.

dtype is the type of the output array. If it is not specified then it will be infered from
the start and stop parameters.

For floating point arguments, the length of the result is ceil((stop - start)/step). Because of 
floating point overflow, this rule may result in the last element of out being greater than stop.
It is better to use linspace for these cases.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html

'''
import numpy as np

return_array = np.arange(5, dtype=float) 
print(return_array)
print(type(return_array))
print(return_array.shape)

return_array = np.arange(0, 10, 2, dtype=int)
print(return_array)
print(type(return_array))
print(return_array.shape)
