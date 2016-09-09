# Ticker Models
import os
import glob
import DataTools


class Price:

    def __init__(self, priceList):

        symbolIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = priceList[symbolIndex]
        date = priceList[dateIndex]
        openPrice = priceList[openIndex]
        high = priceList[highIndex]
        low = priceList[lowIndex]
        close = priceList[closeIndex]
        volume = priceList[volumeIndex]

        self.Symbol = symbol
        self.Date = DataTools.toDate(date)
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

    pricingDictionary = parseFiles(allFiles)

    return pricingDictionary


def parseFiles(files):
    pricingDictionary = dict()
    for file in files:
        #print(file)

        # Read through each line of the file.
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            priceList = line.split(',')

            p = Price(priceList)

            if p.Symbol not in pricingDictionary:
                dateDictionary = dict()
                dateDictionary[p.Date] = p
                pricingDictionary[p.Symbol] = dateDictionary
            else:
                dateDictionary = pricingDictionary[p.Symbol]
                dateDictionary[p.Date] = p

    return pricingDictionary
