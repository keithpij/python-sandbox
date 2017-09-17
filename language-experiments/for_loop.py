''' 
for loop
'''

# Using a for loop to loop through a list.
# Here the varialble emotions is a list and emotion is the iteration variable.
# Also for and in are Python reserved keywords.
print('The for loop.')
emotions = ['Happy', 'Sad', 'Pissed off', 'Grim']
for emotion in emotions:
    print(emotion)
print('End of list.')

# A counting loop.
print('A counting loop.')
count = 0
for n in [1, 4, 6, 8, 12]:
    count = count + 1
print('There are ', count, 'numbers.')

# A summing loop
print('A summing loop.')
total = 0
for n in [1, 4, 6, 8, 12]:
    total = total + n
print('Total: ', total)

# using the range function.
for i in range(1,10):
    print(i)

# Working with numberic lists.
# It may be tempting to write your own loop for determining minimum, maximum, and sum
# but these are builtin functions in python.
numbers = [99, 22, 13, 1, 888]
print(min(numbers))
print(max(numbers))
print(sum(numbers))
