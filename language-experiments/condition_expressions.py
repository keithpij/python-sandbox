'''
Experiments with Conditional Expressions

Conditional expressions take the form:

(expression) if (condition) else (expression)

'''

x=1
y=2

z = 2**(y+1/2) if x+10<0 else 2**(y-1/2)

a = y+1/2
print(a)
print(z)
