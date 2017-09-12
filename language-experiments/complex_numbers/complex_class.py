'''
Basic introduction to complex numbers.
'''

# The following line will print (-9+0j). The imaginary part is preserved even though it is 0.
print(3j**2)

# There has to be a number in front of the j to distinguish between an imaginary number
# and the use of j as a variable.
print(1j)

# Adding a real number to an imaginary number.
print(3+1j)

# Standard operators work with complex numbers.
x = (3+1j) + (10+20j)
print(x)
print(type(x))
print(x.real)
print(x.imag)
