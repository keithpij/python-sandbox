# Market Data
import os
import glob
import datetime

tickerIndex = 0
dateIndex = 1
openIndex = 2
highIndex = 3
lowIndex = 4
closeIndex = 5
volumeIndex = 6


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
    tickerDictionary = dict()
    for file in files:
        print(file)
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            tickerList = line.split(',')

            tickerSymbol = tickerList[0]
            date = tickerList[1]

            if tickerSymbol not in tickerDictionary:
                dateDictionary = dict()
                dateDictionary[date] = tickerList
                tickerDictionary[tickerSymbol] = dateDictionary
            else:
                dateDictionary = tickerDictionary[tickerSymbol]
                dateDictionary[date] = tickerList

    return tickerDictionary


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


def printTickerData(tickerDictionary, tickerName):

    if tickerName.upper() not in tickerDictionary:
        print('No ticker found.')
        return

    print('\n\t\t' + tickerName.upper())
    print('\nDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')

    dateDictionary = tickerDictionary[tickerName.upper()]

    for date in sorted(dateDictionary.keys()):
        dayList = dateDictionary[date]
        change = float(dayList[closeIndex]) - float(dayList[openIndex])
        dateDisplay = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
        d = dateDisplay + '\t'
        d = d + '${:7,.2f}'.format(float(dayList[openIndex])) + '\t'
        d = d + '${:7,.2f}'.format(float(dayList[highIndex])) + '\t'
        d = d + '${:7,.2f}'.format(float(dayList[lowIndex])) + '\t'
        d = d + '${:7,.2f}'.format(float(dayList[closeIndex])) + '\t'
        d = d + '${:7,.2f}'.format(change) + '\t'
        d = d + '{:11,.0f}'.format(float(dayList[volumeIndex]))
        print(d)

    print('\n')


def printLastDateForTicker(marketData, tickerName):

    if tickerName not in marketData:
        return

    daysDict = marketData[tickerName.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    dayList = daysDict[lastDay]

    # Create the date display.
    date = dayList[dateIndex]
    date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
    change = float(dayList[closeIndex]) - float(dayList[openIndex])

    # create the display string
    d = dayList[tickerIndex] + '\t' + date + '\t'
    d = d + '${:7,.2f}'.format(float(dayList[openIndex])) + '\t'
    d = d + '${:7,.2f}'.format(float(dayList[highIndex])) + '\t'
    d = d + '${:7,.2f}'.format(float(dayList[lowIndex])) + '\t'
    d = d + '${:7,.2f}'.format(float(dayList[closeIndex])) + '\t'
    d = d + '${:7,.2f}'.format(change) + '\t'
    d = d + '{:11,.0f}'.format(int(dayList[volumeIndex]))
    print(d)

# Main execution

# Load files.
print('Loading Files ...')
marketData = filesToList()
print(str(len(marketData)) + ' items.')

today = datetime.date.today()
print('\nToday\'s date: ' + str(today) + '\n')

# Print the last pricing information for each ticker in the portfolio
print('Ticker\tDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')
portfolio = getPortfolio()
for ticker in portfolio:
    printLastDateForTicker(marketData, ticker)

# User input
while True:
    tickerName = raw_input('Enter a ticker symbol:  ')

    if tickerName == 'quit':
        break

    printTickerData(marketData, tickerName)
