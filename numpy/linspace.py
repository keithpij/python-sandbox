'''
numpy.linespace experiments

Useage:
    numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)

Description:
    Returns num evenly spaced samples, calculated over the interval [start, stop].

Parameters:
    start: scalar
    The start value of the sequence.
    
    stop: scalar
    The end value of the sequence, unless endpoint is set to False.
    The end value of the sequence, unless endpoint is set to False. In that case, 
    the sequence consists of all but the last of num + 1 evenly spaced samples, 
    so that stop is excluded. Note that the step size changes when endpoint is False.
    
    num: int, optional
    The number of samples to generate.
    
    endpoint: bool, optional
    If True, stop is the last sample. Otherwise, it is not included. Default is True.
    
    retstep: bool, optional
    If True, return (samples, step), where step is the spacing between samples.

    dtype : dtype, optional
    The type of the output array. If dtype is not given, infer the data type from the other input arguments.
    New in version 1.9.0.

Returns:	
    samples : ndarray
    There are num equally spaced samples in the closed interval [start, stop] or the half-open interval [start, stop) (depending on whether endpoint is True or False).

    step : float, optional
    Only returned if retstep is True
    Size of spacing between samples.

'''
import numpy as np

values, step = np.linspace(0, 5, 100, retstep=True)
print(values)
print(step)
