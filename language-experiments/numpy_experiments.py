''' Experiments with numpy. '''
import numpy as np


coefficient_matrix = np.array([[2, -1],
                               [-3, -2]], float)

right_side_vector = np.array([[-4],
                              [-1]], float)

# The system has an exact answer so we can use solve.
unknown_vector = np.linalg.solve(coefficient_matrix, right_side_vector)
print(unknown_vector)

# check the answer
print(np.dot(coefficient_matrix, unknown_vector))

# You can add a singe value to a matrix.
# Here 5 will be added to every value in the matrix.
print(coefficient_matrix + 5)

another_matrix = np.array([[2, -1],
                           [-3, -2]], float)

x = another_matrix == coefficient_matrix
# You cannot compare numpy arrays using the equivalence operator.
print(x)
print(x.shape)

# assert will not work because the result of the equivalence check is a matrix itself.
#assert another_matrix == coefficient_matrix
