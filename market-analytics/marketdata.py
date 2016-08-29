# Market Data
import datetime
import os
import TickerModels
import IndexModels
import sys


def loadPortfolio():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data')
    portfolioFile = os.path.join(pricingDir, 'portfolio.txt')
    portfolio = []
    fhand = open(portfolioFile)
    for ticker in fhand:
        ticker = ticker.strip()
        portfolio.append(ticker)

    return portfolio


def printPortfolio():
    today = datetime.date.today()
    print('\nToday\'s date: ' + str(today) + '\n')

    # Print the last pricing information for each ticker in the portfolio
    print('Ticker\tDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')
    # Print the indecies.
    IndexModels.PrintLastDateForIndex(indexDictionary, 'DJI')
    IndexModels.PrintLastDateForIndex(indexDictionary, 'NAST')

    # Loop through the portfolio.
    portfolio = loadPortfolio()
    for ticker in portfolio:
        printLastDateForTicker(tickerDictionary, ticker)


def printTickerData(tickerName, dailyPricingDict):

    print('\n\t\t' + tickerName.upper())
    print('\nDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')

    for date in sorted(dailyPricingDict.keys()):
        t = dailyPricingDict[date]
        d = str(date) + '\t'
        d = d + '${:7,.2f}'.format(t.Open) + '\t'
        d = d + '${:7,.2f}'.format(t.High) + '\t'
        d = d + '${:7,.2f}'.format(t.Low) + '\t'
        d = d + '${:7,.2f}'.format(t.Close) + '\t'
        d = d + '${:7,.2f}'.format(t.Change) + '\t'
        d = d + '{:11,.0f}'.format(t.Volume)
        print(d)

    print('\n')


def printLastDateForTicker(marketData, tickerName):
    if tickerName.upper() not in tickerDictionary:
        return

    daysDict = marketData[tickerName.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    t = daysDict[lastDay]

    # create the display string
    d = t.Symbol + '\t' + str(t.Date) + '\t'
    d = d + '${:7,.2f}'.format(float(t.Open)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.High)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.Low)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.Close)) + '\t'
    d = d + '${:7,.2f}'.format(t.Change) + '\t'
    d = d + '{:11,.0f}'.format(int(t.Volume))
    print(d)


def getDateRange(dailyPricingDict, daysInPast):
    endDate = datetime.date.today()
    daysInPastDelta = datetime.timedelta(days=daysInPast)
    startDate = endDate - daysInPastDelta
    r = dict()
    for date in sorted(dailyPricingDict.keys()):
        if date >= startDate and date <= endDate:
            r[date] = dailyPricingDict[date]
    return r

def findWinners():
    r = dict()
    for t in tickerDictionary.keys():
        dailyPricingDict = tickerDictionary[t]
        dateList = sorted(dailyPricingDict.keys())
        firstKey = dateList[0]
        lastKey = dateList[-1]
        first = dailyPricingDict[firstKey]
        last = dailyPricingDict[lastKey]
        g = (last.Close - first.Open) / first.Open
        if first.Open > 25.0 and last.Close < 75 and g > 0.25 and g < 0.5:
            r[t] = g
    return r


def printWinners(w):
    count = 0
    for t in w.keys():
        count = count + 1
        print(t + ' ' + str(w[t]))
    print(str(count) + ' companies.')

def getUserRequests():
    # User request loop
    while True:
        command = input('--> ')

        # Quit the input loop.
        if command == '-q':
            break

        if command == '-h':
            showHelp()

        if command == '-w':
            winners = findWinners()
            printWinners(winners)

        # Search by name.  The search text must be a regular expression.
        if command[0:2] == '-t':
            params = command.split(' ')
            t = params[0][3:].upper()

            # Check and see if ticker symbol exists.
            if t not in tickerDictionary:
                print('Ticker ' + t + ' not found.')
                continue

            # All the pricing info for the specified ticker.
            dailyPricingDict = tickerDictionary[t]

            # No days in the past specified.
            if len(params) == 1:
                printTickerData(t, dailyPricingDict)
            elif params[1][0:2] == '-d':
                d = params[1][3:]  # days in the past
                d = int(d)
                dailyPricingDict = getDateRange(dailyPricingDict, d)
                printTickerData(t, dailyPricingDict)
            else:
                print('Unrecogonized parameter ' + param(1)[0:2])

        if command[0:2] == '-p':
            printPortfolio()


# Main execution
# Passed arguments
print(sys.argv)

# Load ticker and index files.
print('Loading Data ...')
tickerDictionary = TickerModels.LoadFiles()
indexDictionary = IndexModels.LoadFiles()

print(str(len(tickerDictionary)) + ' items loaded.')

# This function is a user request loop.
getUserRequests()
