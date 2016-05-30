# Tuples

# A tuple is a list of values.
workout = ('Chest Press', 'Bicep Curls', 'Shoulder Press', 'Sit ups')
print(type(workout))
print(workout)

notATuple = ('Leg Curls')
print(type(notATuple))
print(notATuple)

# If there is only one value you need the trailing comma.
isATuple = ('Leg Curls',)
print(type(isATuple))
print(isATuple)

emptyTuple = tuple()
print(emptyTuple)

tupleOfCharacters = tuple('alongword')
print(tupleOfCharacters)

# Most list operators work on tuples.
print(tupleOfCharacters[0])
print(tupleOfCharacters[2:5])
print(workout[1])

newTuple = ('z',) + tupleOfCharacters[2:5]
print(newTuple)

# Note:  Unsaved comment - The GitHub Desktop client is smart enough to save your files before switching branches.
