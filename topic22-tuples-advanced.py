# Advanced tuple techniques

# Operators and tuples.
t1 = (0, 1, 2, 3)
t2 = (1, 2, 3, 4)
t3 = (0, 1, 2, 3)
t4 = (0, 1, 2, 4)
t5 = (4, 3, 2, 1)
print(t1 < t2)
print(t1 == t3)
print(t1 is t3)

# Note: Tuples do not have a sort method to sort the values.
# print(t5.sort())

# Sorting tuples
l = list()
l.append(t1)
l.append(t2)
l.append(t3)
l.append(t4)
l.append(t5)
l.sort()
print(l)
l.sort(reverse=True)
print(l)

# Multiple assignment from a list of values.
# This will also work if w is a tuple.
w = ['Chest Press', 'Arm Curls', 'Pull downs']
e1, e2, e3 = w
print(e1)
print(e2)
print(e3)

# Dictionaries, Lists, and Tuples.
d = {'APPL': 'Apple', 'GOOG': 'Alphabet', 'ORCL': 'Oracle'}
listOfTuples = d.items()
print(listOfTuples)
listOfTuples.sort()
print(listOfTuples)

# Using a tuple to loop through the key value pairs within a dictionary.
for key, val in d.items():
    print(key, val)

# Using tuples as dictionary keys.
marketData = dict()
marketData['MSFT', '20160516'] = 51.83
marketData['MSFT', '20160517'] = 50.51
marketData['MSFT', '20160518'] = 50.81
marketData['MSFT', '20160519'] = 50.32
marketData['MSFT', '20160520'] = 50.62

for ticker, date in marketData:
    print ticker, date, marketData[ticker, date]


def someFunction(a, b):
    return a, b

returnTuple = someFunction(3, 4)
print(returnTuple)