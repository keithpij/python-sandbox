''' Experiments with the numpy dot function. '''
import numpy as np

# Create a 2 x 2 matrix.
matrix = np.array([[1, 1],
                               [1, 1]], int)


vector = np.array([[2],
                   [3]], int)

print('The shape of matrix is: ' + str(matrix.shape))
print('The value of matrix is: ')
print(matrix)

print(vector.shape)
print(vector)

dot_product = np.dot(vector, matrix)

print(dot_product.shape)
print(dot_product)
