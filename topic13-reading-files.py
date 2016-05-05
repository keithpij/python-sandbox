# Reading text files

# Function definitions
fileName = 'eod-data/NASDAQ_20160422.txt'

# This is another way to specify the file.
# fileName = './eod-data/NASDAQ_20160422.txt'

# This will not work.
# fileName = '/eod-data/NASDAQ_20160422.txt'


# Function to open a file and print each line as it appears in the file.
# This function will also count and print the total number of lines.
def printFile(fileName):
    count = 0

    fhand = open(fileName)
    for line in fhand:
        count = count + 1
        print(line)

    display = str(count) + ' lines'
    print(display)


def searchFile(fileName, searchTicker):
    fhand = open(fileName)
    for line in fhand:
        if line.startswith(searchTicker.upper()):
            data = line.split(',')

            display = data[0] + ' ' + data[1] + ' ' + data[2] + ' ' + data[3]
            display = display + ' ' + data[4] + ' ' + data[5] + ' ' + data[6]
            display = display + ' ' + data[7]

            print(display)
            print(len(data))
            break


# Main - execution startes here.
# Function definitions must be above.

# printFile(fileName)

ticker = raw_input('Enter a ticker: ')
searchFile(fileName, ticker)
