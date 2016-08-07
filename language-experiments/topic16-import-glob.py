# Importing glob
import os
import glob

# Create the search parameter for the glob function.
# It is important to use the join function instead of hardcoding forward
# slashes and back slashes in the code.  This makes sure the code will run
# on both Windows and Mac.
currentWorkingDir = os.getcwd()
pricingDir = os.path.join(currentWorkingDir, 'eod-data')
pricingDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt')

files = glob.glob(pricingDirSearch)

for file in files:
    print(file)
