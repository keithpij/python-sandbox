# Lists

# Creating a list.
myMenu = ['Steak', 'Fish', 'Sushi', 'Stew', 'Chili']
print(myMenu)
print(type(myMenu))

# Referencing list items.
print(myMenu[0])
print(myMenu[-1])

# The len() function with lists.
print('The number of items in myMenu is: ' + str(len(myMenu)))

# Lists are mutable.
myMenu[0] = 'Chicken'
print(myMenu)

# Looping through lists.
for food in myMenu:
    print(food)

for i in range(len(myMenu)):
    myMenu[i] = 'Good ' + myMenu[i]
print(myMenu)

# Empty lists
empty = []
for item in empty:
    print('You will never get here.')

# List operators
# Addition
numbers1 = [1, 2, 3]
numbers2 = [4, 5, 6]
sumOfNumbers = numbers1 + numbers2
print(sumOfNumbers)

# Multiplication
print(numbers1 * 3)

# List slices
print(sumOfNumbers[1:3])
print(sumOfNumbers[:4])
sumOfNumbers[2:4] = [0, 0]
print(sumOfNumbers)

# List methods
# append
todo1 = ['Make Dinner', 'Workout']
todo1.append('Study Python')
print(todo1)

# extend
todo2 = ['Clean office', 'Mow the lawn', 'Pick weeds']
todo1.extend(todo2)
print(todo1)

# sort
todo1.sort()
print(todo1)

# pop
task = todo1.pop(0)
print(task)
print(todo1)

# del
# del can also be used with slices
del(todo1[1])
print(todo1)

# remove
todo1.remove('Workout')
print(todo1)

# functions
print('Length: ' + str(len(todo1)))
print('Max: ' + max(todo1))
print('Min: ' + min(todo1))

nums = [1, 2, 3, 4, 5, 6, 7]
print('Sum: ' + str(sum(nums)))

# lists
longWord = 'supercalifragilisticexpialidosous'
l = list(longWord)
print('A long word converted to a list.')
print(l)

# split
sentence = 'This is a sentence.'
print('Splitting a sentence.')
l = sentence.split()
print(l)

# join
delimiter = '-'
s = delimiter.join(l)
print('Joining the items in a list.')
print(s)

# aliasing a list
list1 = [1, 2, 3]
list2 = [1, 2, 3]
list3 = list1
string1 = 'Guitar'
string2 = 'Guitar'
print(list1 is list2)
print(string1 is string2)
print(list1 is list3)
