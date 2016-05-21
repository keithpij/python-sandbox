# String manipulation

print('String Manipulation')

print('Getting the first character of a string.')
drink = 'Single Malt Scotch'
character = drink[0]
print(character)

print('Determining the length of a string.')
print(len(drink))

print('Getting the last character of a string.')
lastCharacter = drink[len(drink)-1]
print(lastCharacter)

print('Looping through the characters in a string.')
for c in drink:
    print(c)

print('Another way to loop through the characters in a string.')
i = 0
while i < len(drink):
    c = drink[i]
    print(c)
    i = i + 1

print('String slices')
print(drink[0:6])
print(drink[7:11])
print(drink[12:19])
print(len(drink))

print('Updating strings')
# The statement below will cause an error because strings are immutable.
#drink[6] = "-"
newDrink = drink[0:6] + '-' + drink[7:11] + '-' + drink[12:19]
print(newDrink)

print('The in operator')
print(' ' in drink)

# This will output a list of all methods for strings.
print('String methods')
print(dir(drink))

uppercaseDrink = drink.upper()
print(uppercaseDrink)

index = drink.find(' ')
print index

day = 'AAPL,20160415,112.11,112.3,109.73,109.85,46938900'
index = day.find(',')
ticker = day[0:index]
print(ticker)
print(len(ticker))

# This is best done in interactive mode.  It will produce help text on the
# specified function.
# help(drink.split)

# Uppercase letters come before lowercase letters.
print('String comparison')
if 'A' < 'a':
    print('A is less than a')
elif 'A' > 'a':
    print('A is greater than a')
else:
    print('A is equal to a')

import string
print(string.punctuation)
stringWithPuncuations = 'a-b,c!;:'
print(stringWithPuncuations)
print(stringWithPuncuations.translate(None, string.punctuation))
