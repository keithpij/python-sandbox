# Escape Characters

print('\\ Printing a backslash.')

print('\' Printing a single-quote.')

print('\" Printing a double-quote backslash.')

print('\a ASCII bell.')

# Notice that when this runs the '-' is gone.
print('ASCII back-\bspace.  X')

print('Form Feed\fForm Feed')

print('Line feed\nNext line.')

print('Portions of this line will be overwritten.\rCarriage return.')

# TODO - this is not working.
print('Character with octal value of 16:  \o17')

# This is an infinite loop.  Control-c will stop the infinite loop.
while True:
    for i in ["/", "-", "|", "\\", "|"]:
        print "%s\r" % i,
