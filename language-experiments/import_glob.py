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


def search_files(dir):
    file_list = []
    extension_coounts = dict()
    for root, dirs, files in os.walk(dir):
        for name in files:
            file_list.append(os.path.join(root, name))
            extension = os.path.splitext(name)[1]
            if extension in extension_coounts:
                extension_coounts[extension] += 1
            else:
                extension_coounts[extension] = 1

    return file_list, extension_coounts


if __name__ == '__main__':
    file_list, extension_counts = search_files('/Users/keithpij/code')
    for extension in extension_counts:
        print(extension + ':  ' + str(extension_counts[extension]))
    