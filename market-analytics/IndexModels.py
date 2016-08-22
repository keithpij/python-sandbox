# Index Models
import os
import glob


class Index:

    def __init__(self, tickerList):

        nameIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = tickerList[nameIndex]
        date = tickerList[dateIndex]
        openPrice = tickerList[openIndex]
        high = tickerList[highIndex]
        low = tickerList[lowIndex]
        close = tickerList[closeIndex]
        volume = tickerList[volumeIndex]

        self.Symbol = symbol
        self.Date = date
        self.Open = float(openPrice)
        self.High = float(high)
        self.Low = float(low)
        self.Close = float(close)
        self.Change = self.Close - self.Open
        self.Volume = int(volume)


def LoadFiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data')
    indexDirSearch = os.path.join(pricingDir, 'INDEX*.txt')

    # Notice how two lists can be added together.
    indexFiles = glob.glob(indexDirSearch)

    indexDictionary = parseFiles(indexFiles)

    return indexDictionary


def parseFiles(files):
    indexDictionary = dict()
    for file in files:
        print(file)

        # Read through each line of the file.
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            indexList = line.split(',')

            # Create an instance of the Index class.
            i = Index(indexList)

            # Only interested in Dow Jones and NASDAQ.
            if i.Symbol != 'DJI' and i.Symbol != 'NAST':
                continue

            if i.Symbol not in indexDictionary:
                dateDictionary = dict()
                dateDictionary[i.Date] = i
                indexDictionary[i.Symbol] = dateDictionary
            else:
                dateDictionary = indexDictionary[i.Symbol]
                dateDictionary[i.Date] = i

    return indexDictionary


def PrintLastDateForIndex(marketData, tickerName):

    if tickerName not in marketData:
        return

    daysDict = marketData[tickerName.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    t = daysDict[lastDay]

    # Create the date display.
    date = t.Date
    date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
    change = float(t.Close) - float(t.Open)

    # create the display string
    d = t.Symbol + '\t' + date + '\t'
    d = d + '{:7,.2f}'.format(float(t.Open)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.High)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Low)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Close)) + '\t'
    d = d + '{:7,.2f}'.format(change) + '\t'
    d = d + '{:11,.0f}'.format(int(t.Volume))
    print(d)
