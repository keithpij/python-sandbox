# Ticker Models
import os
import glob


class Ticker:

    def __init__(self, tickerList):

        tickerIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = tickerList[tickerIndex]
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
    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt')

    # Notice how two lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    allFiles = nasdaqFiles + nyseFiles

    tickerDictionary = parseFiles(allFiles)

    return tickerDictionary


def parseFiles(files):
    tickerDictionary = dict()
    for file in files:
        print(file)

        # Read through each line of the file.
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            tickerList = line.split(',')

            t = Ticker(tickerList)
            # if close > 100:
            #    print(t.Symbol)

            if t.Symbol not in tickerDictionary:
                dateDictionary = dict()
                dateDictionary[t.Date] = t
                tickerDictionary[t.Symbol] = dateDictionary
            else:
                dateDictionary = tickerDictionary[t.Symbol]
                dateDictionary[t.Date] = t

    return tickerDictionary
