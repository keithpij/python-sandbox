'''
Experiments with comprehensions.
'''

# Set comprehensions
y = {2*x for x in {1,2,3,4}}
print(y)

# A comprehension with a filter.
y = {2*x for x in {1,2,3,4} if x>2}
print(y)

# Comprehension with a union.
S = {1,2,3}
y = {x*x for x in S | {5,7}}
print(type(y))
print(y)

# Comprehension that produces a list.
S = {1,2,3}
y = [x*x for x in {5,7}]
print(type(y))
print(y)

