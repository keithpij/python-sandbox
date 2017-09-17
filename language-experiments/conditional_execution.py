# Conditional execution

x = 1
y = 2
z = 3

# A basic if statement
if (x > 0):
    print('The value of x is greater than 0.')

# A basic if-else statement.
if (x < y):
    print('The value of x is less than the value of y.')
else:
    print('The value of x is not less than the value of y.')

# Chained conditional statements using the elif keyword.
if (x < y):
    print('The value of x is less than the value of y.')
elif (x > y):
    print('The value of x is greater than the value of y.')
else:
    print('The values of x and y are equal.')

# The pass statement.
if x is z:
    pass
else:
    print('The variable x does not point to the same value as z.')

# The Guardian pattern.
# The second condition is never executed.
if (x > y) and (w == z):
    print('The code will never get here.')
else:
    print('This will not cause an undefined error.')

# Nested conditions.
if (x > 0):
    if (x < 10):
        print('The value of x is between 0 and 10.')

# This will produce the same result as the nested if above.
if (x > 0) and (x < 10):
    print('The value of x is between 0 and 10.')
