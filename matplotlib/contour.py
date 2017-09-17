'''
matplotlib.pyplot.contour experiments

A contour line of a function with two variables is a curve which connects points with the same values. An example
of a countour line in real life would be the lines that show equal evelvation in a topological map. Like a topologial
map any given function may be more than one curve.

It is important to note that the coutour function does not evaluate functions directly. You do not pass it a function
reference and ask it to do some analysis and plot lines. It is up to you - the engineer - to generate a meshgrid of 
X values, Y values, and then run the function with your meshgrid to get the results which in the examples that follow
will be known as Z values.

http://www.python-course.eu/matplotlib_contour_plot.php


ndarray.shape
Tuple of array dimensions.  May be used to “reshape” the array, as long as this would not 
require a change in the total number of elements.
https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html


numpy.zeros(shape, dtype=float, order='C')
Return a new array of given shape and type, filled with zeros.

Parameters:	
    shape : int or sequence of ints
    Shape of the new array, e.g., (2, 3) or 2.
    dtype : data-type, optional
    The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
    order : {‘C’, ‘F’}, optional
    Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) 
    order in memory.

Returns:	
    out : ndarray
    Array of zeros with the given shape, dtype, and order.

https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html

'''

import matplotlib.pyplot as plt
import numpy as np
import random


def step_function_example():
    x_values = np.linspace(0, 4, 5)
    y_values = np.linspace(0, 4, 5)
    X, Y = np.meshgrid(x_values, y_values)
    
    def step_function(x,y):
        if x == y: return 0
        else: return random.randint(1, 1000000)

    Z = np.zeros(X.shape)
    # We need the x values and the y values produced by meshgrid.
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
            x = X[i,j]
            y = Y[i,j]
            z = step_function(x,y)
            Z[i,j] = z

    return X, Y, Z


def sqrt_example():

    # First we generate array of all the X values we would like to consider as well as all the 
    # Y values we would like to consider.
    xlist = np.linspace(-3.0, 3.0, 3)
    ylist = np.linspace(-3.0, 3.0, 4)

    X, Y = np.meshgrid(xlist, ylist)

    Z = np.sqrt(X**2 + Y**2)
    return X, Y, Z


def plot_contour(X, Y, Z):
    plt.figure()

    cp = plt.contour(X, Y, Z, alpha=0.4)

    plt.clabel(cp, inline=True, fontsize=10)
    plt.title('Contour Plot')
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.show()


def custom_function(x, y):
    weight_y = .5
    weight_x = .5

    z = (weight_x * x) + (weight_y * y) 
    return z


if __name__ == '__main__':

    X, Y, Z = step_function_example()
    plot_contour(X, Y, Z)

    #print(X.shape)
    #print(Y.shape)
    #print(Z.shape)

    #print(X)
    #print(Y)
    #print(Z)

