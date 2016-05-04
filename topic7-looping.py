# Looping in Python

# TODO - Determine what you have to do to bust out of an infinite loop
# should you accidentally get into one from bash.

# Using the while loop to create a loop with a specific number of iterations.
n = 5
while n > 0:
    print(n)
    n = n - 1
print('Boom')

# Using the break command within an infinite loop.
# Note:  'True' must start with a capital 'T'
print("The echo loop")
while True:
    line = raw_input('> ')
    if line == 'done':
        break
    print(line)
print('The loop is done.')

# Using the continue command within an infinit loop.
print('The break and continue loop.')
print('Enter "done" to exit the loop.')
print('Enter "#" to skip current iteration but continue looping.')
while 'True':
    line = raw_input('> ')
    if line[0] == "#":
        print('The loop will continue.')
        continue
    if line == 'done':
        print('The loop is about to end.')
        break
    print(line)
print('The Loop is complete.')

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

# Working with numberic lists.
# It may be tempting to write your own loop for determining minimum, maximum, and sum
# but these are builtin functions in python.
numbers = [99, 22, 13, 1, 888]
print(min(numbers))
print(max(numbers))
print(sum(numbers))
