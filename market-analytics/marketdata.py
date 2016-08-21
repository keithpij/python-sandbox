# Market Data
import datetime
import os
import TickerModels
import IndexModels


def getPortfolio():
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


def printTickerData(tickerDictionary, tickerName):

    if tickerName.upper() not in tickerDictionary:
        print('No ticker found.')
        return

    print('\n\t\t' + tickerName.upper())
    print('\nDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')

    dateDictionary = tickerDictionary[tickerName.upper()]

    for date in sorted(dateDictionary.keys()):
        t = dateDictionary[date]
        dateDisplay = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
        d = dateDisplay + '\t'
        d = d + '${:7,.2f}'.format(t.Open) + '\t'
        d = d + '${:7,.2f}'.format(t.High) + '\t'
        d = d + '${:7,.2f}'.format(t.Low) + '\t'
        d = d + '${:7,.2f}'.format(t.Close) + '\t'
        d = d + '${:7,.2f}'.format(t.Change) + '\t'
        d = d + '{:11,.0f}'.format(t.Volume)
        print(d)

    print('\n')


def printLastDateForTicker(marketData, tickerName):

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
    d = d + '${:7,.2f}'.format(float(t.Open)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.High)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.Low)) + '\t'
    d = d + '${:7,.2f}'.format(float(t.Close)) + '\t'
    d = d + '${:7,.2f}'.format(change) + '\t'
    d = d + '{:11,.0f}'.format(int(t.Volume))
    print(d)


# Main execution

# Load files.
print('Loading Data ...')
tickerDictionary = TickerModels.LoadFiles()
indexDictionary = IndexModels.LoadFiles()

print(str(len(tickerDictionary)) + ' items.')

today = datetime.date.today()
print('\nToday\'s date: ' + str(today) + '\n')

# Print the last pricing information for each ticker in the portfolio
print('Ticker\tDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')
# Print the indecies.
IndexModels.PrintLastDateForIndex(indexDictionary, 'DJI')
IndexModels.PrintLastDateForIndex(indexDictionary, 'NAST')

# Loop through the portfolio.
portfolio = getPortfolio()
for ticker in portfolio:
    printLastDateForTicker(tickerDictionary, ticker)

# User request
while True:
    tickerName = input('Enter a ticker symbol:  ')

    if tickerName == 'quit':
        break

    printTickerData(tickerDictionary, tickerName)
