# Dictionaries
print('Fun with Dictionaries')

# Create an empty dictionary.
tickersDict = dict()
print(tickersDict)

# This will output a list of all methods for strings.
print('Dictionary methods')
print(dir(tickersDict))

tickersDict['AAPL'] = 'Apple'
tickersDict['GOOG'] = 'Alphabet'
print(tickersDict)
msg = 'The number of key-value pairs in tickersDict is:  '
print(msg + str(len(tickersDict)))

# The in operator
# Python uses hash tables for dictionaries.
print('AAPL' in tickersDict)
print('MSFT' in tickersDict)

# The values method returns a list.
vals = tickersDict.values()
'Apple' in vals

# Sorting dictionaries.
d = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
print(d)

# Sort the keys
print(sorted(d))

# Sort the values
print(sorted(d.values()))

# Sort the keys by the values.
print(sorted(d, key=d.__getitem__))

# d.get returns the value at the first parameter location
# or if it does not exist the second parameter is the default value.
longword = 'Supercalifragilisticexpialidocious'
d = dict()
count = 0
for character in longword:
    count = count + 1
    d[character] = d.get(character, 0) + 1
print(longword)
print(count)
print(d)

for key in d:
    print(key)

keysAsList = d.keys()
print(keysAsList)
keysAsList.sort()
for k in keysAsList:
    print(k, d[k])

d = dict()
f = dict()
f['description'] = 'Test ticket - Eileen'
f['summary'] = 'Test Story Ticket'
f['project'] = 'TEST'
f['assignee'] = 'zkaqo35'
f['type'] = 'story'
f['trackerid'] = 'qzap://QzTracker/ticket/000-00000'
d['fields'] = f

print(d)
print(type(d))
