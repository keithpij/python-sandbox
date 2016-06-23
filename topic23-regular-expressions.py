# Regular Expressions
import re

# Basic search.
line = 'xxtokenxxxxxxx'
print(line)
if re.search('token', line):
    print('token found')
else:
    print('token not found')

# Using the '^' so specify that the search string must be at
# the beginning of the string being searched.
line = 'xtoken xxxxxxxxx'
print(line)
if re.search('^token', line):
    print('token found')
else:
    print('token not found')

# Using '.' to indicate that any character is a match.
line = 'xtoken xxxxxxxxx'
print(line)
if re.search('.token', line):
    print('token found')
else:
    print('token not found')

# using '+' to indicate 1 or more of the previous character which in this
# case is a '.' indicating that any character is a match.
line = 'xtoken xxxxxxxxx'
# line = 'xtoken'  # This will not result in a match.
print(line)
if re.search('.token.+', line):
    print('token found')
else:
    print('token not found')

# using '*' to indicate 0 or more of the previous character which in this
# case is a '.' indicating that any character is a match.
# line = 'xtoken xxxxxxxxx'
line = 'xtoken'  # This will result in a match because * means 0 or more.
print(line)
if re.search('.token.*', line):
    print('token found')
else:
    print('token not found')

# '\S' matches a non-whitespace character.
line = 'xxxx <keith@codeclimate.com>; <keithpij@msn.com>; <keithpij@gmail.com> yyyy'
emails = re.findall('\S+@\S+', line)
print(emails)

# Nnow we are looking for substrings that start with a single lower
# case letter, uppercase letter, or a number.
emails = re.findall('[a-zA-Z0-9]\S*@\S*[a-zA-Z]', line)
print(emails)
