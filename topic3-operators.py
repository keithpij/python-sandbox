# Operators and operands

# Operators in python are:  +, -, *, /, **, %   addition, subtraction,
# multiplication, division, and exponentiation, and modulus.
# Operands are the values that the operators are applied to.

print('Addition')
s = 3 + 5
print(s)

print('Subtraction')
a = 10
b = 3.3
c = a - b
print(c)

print('Multiplication')
m = a*b
print(m)

# Notice that if both numbers are integers then the result will be an integer.
print('Division')
print('50 / 13')
b = 50/13
print(b)

# Notice that if one number is a float then the result will be a float.
print('Division')
print('50 / 13')
b = 50/13.2
print(b)

print('Exponentiation')
x = 2
e = 3
r = x**e
print(r)

print('Modulus')
n = 10
m = 3
remainder = n % m
print(remainder)

print('Order of operations')
# Order of operations:
# 1. Parentheses
# 2. Exponentiation
# 3. Multiplication and Division
# 4. Addition and Subtraction
print('(1+1) * (2+2)')
r = (1+1) * (2+2)
print(r)

print('3**2+1')
r = 3**2+1
print(r)

print('3*4-1')
r = 3*4-1
print(r)
