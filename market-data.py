# Market Data
import os
import glob


def filesToList():
    currentWorkingDir = os.getcwd()
    pricingDir = os.path.join(currentWorkingDir, 'eod-data')
    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt')

    # Notice how two lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    allFiles = nasdaqFiles + nyseFiles

    marketList = parseFiles(allFiles)

    return marketList


def parseFiles(files):
    r = []
    for file in files:
        print(file)
        fhand = open(file)
        for line in fhand:
            line = line.strip()
            tickerData = line.split(',')
            r.append(tickerData)

    return r


def getPortfolio():
    currentWorkingDir = os.getcwd()
    pricingDir = os.path.join(currentWorkingDir, 'eod-data')
    portfolioFile = os.path.join(pricingDir, 'portfolio.txt')
    portfolio = []
    fhand = open(portfolioFile)
    for ticker in fhand:
        ticker = ticker.strip()
        portfolio.append(ticker)

    return portfolio


def printTickerData(marketData, tickerName):
    tickerIndex = 0
    dateIndex = 1
    openIndex = 2
    highIndex = 3
    lowIndex = 4
    closeIndex = 5
    volumeIndex = 6

    print('\n\t\t' + tickerName.upper())
    print('\nDate\t\tOpen\tHigh\tLow\tClose\tVolume')

    for tickerList in marketData:
        if tickerName.upper() == tickerList[tickerIndex]:
            date = tickerList[dateIndex]
            date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
            display = date + '\t'
            display = display + tickerList[openIndex] + '\t'
            display = display + tickerList[highIndex] + '\t'
            display = display + tickerList[lowIndex] + '\t'
            display = display + tickerList[closeIndex] + '\t'
            display = display + tickerList[volumeIndex]
            print(display)

    print('\n')


def printLastDateForTicker(marketData, tickerName):
    tickerIndex = 0
    dateIndex = 1
    openIndex = 2
    highIndex = 3
    lowIndex = 4
    closeIndex = 5
    volumeIndex = 6

    for tickerList in marketData:
        if tickerName.upper() == tickerList[tickerIndex]:
            saveTickerList = tickerList

    # Create the date display.
    date = tickerList[dateIndex]
    date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]

    # create the display string
    display = tickerList[tickerIndex] + '\t' + date + '\t'
    display = display + tickerList[openIndex] + '\t'
    display = display + tickerList[highIndex] + '\t'
    display = display + tickerList[lowIndex] + '\t'
    display = display + tickerList[closeIndex] + '\t'
    display = display + tickerList[volumeIndex]
    print(display)
    print('\n')

# Main execution

# Load files.
print('Loading Files ...')
marketData = filesToList()
print(str(len(marketData)) + ' items.')

# Print the last pricing information for each ticker in the portfolio
print('Ticker\tDate\t\tOpen\tHigh\tLow\tClose\tVolume')
portfolio = getPortfolio()
for ticker in portfolio:
    printLastDateForTicker(marketData, ticker)

# User input
while True:
    tickerName = raw_input('Enter a ticker symbol:  ')

    if tickerName == 'quit':
        break

    printTickerData(marketData, tickerName)
