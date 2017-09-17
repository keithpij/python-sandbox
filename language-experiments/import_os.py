# Imported os utilities
import os

currentWorkingDir = os.getcwd()
print(currentWorkingDir)

# It is important to use os.path.join becuase Windows and Mac have a different
# way of constructing file paths.  Windows uses a backslash '\' and file paths
# Linux and Apple use a forward slash '/'.  os.path.join will do the right
# thing based on the current platform.
pricingDir = os.path.join(currentWorkingDir, 'eod-data')
print(pricingDir)

count = 0
for (dirName, dirs, files) in os.walk(pricingDir):
    for fileName in files:
        count = count + 1
        print(fileName)

print 'Files: ', count
