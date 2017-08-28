''' 
while loop
'''

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

