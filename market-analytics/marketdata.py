# Market Data
import datetime
import os
import sys
import price
import index
import portfolio
import company
import search


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
    for s in pricingDictionary.keys():
        dailyPricingDict = pricingDictionary[s]
        dateList = sorted(dailyPricingDict.keys())
        firstKey = dateList[0]
        lastKey = dateList[-1]
        first = dailyPricingDict[firstKey]
        last = dailyPricingDict[lastKey]
        g = (last.Close - first.Open) / first.Open
        if first.Open > 25.0 and last.Close < 75 and g > 0.25 and g < 0.5:
            r[s] = g
    return r


def printWinners(w):
    count = 0
    for s in w.keys():
        count = count + 1
        print(s + ' ' + str(w[s]))
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
        if command[0:2] == '-s':
            params = command.split(' ')
            s = params[0][3:].upper()

            # Check and see if ticker symbol exists.
            if s not in pricingDictionary:
                print('Company ' + s + ' not found.')
                continue

            # All the pricing info for the specified ticker.
            dailyPricingDict = pricingDictionary[s]

            # No days in the past specified.
            if len(params) == 1:
                printTickerData(s, dailyPricingDict)
            elif params[1][0:2] == '-d':
                d = params[1][3:]  # days in the past
                d = int(d)
                dailyPricingDict = getDateRange(dailyPricingDict, d)
                printTickerData(s, dailyPricingDict)
            else:
                print('Unrecogonized parameter ' + param(1)[0:2])

        if command[0:2] == '-p':
            portfolio.printPortfolio(companyDictionary, indexDictionary, pricingDictionary)


if __name__ == '__main__':
    # Main execution
    # Passed arguments
    print(sys.argv)

    # Start off with a date range that is 30 days in the past.
    enddate = datetime.date.today()
    daysInPastDelta = datetime.timedelta(30)
    startdate = enddate - daysInPastDelta
    searchcriteria = search.Criteria(startdate, enddate)

    # Load company files, company pricing files and index files.
    print('Loading Data ...')
    '''
    companyDictionary, count = company.loaddatafromfiles()
    pricingDictionary = price.loaddatafromfiles()
    indexDictionary = index.loaddatafromfiles()
    '''
    companyDictionary, count = company.loaddatafromblobs()
    pricingDictionary = price.loaddatafromblobs()
    indexDictionary = index.loaddatafromblobs()

    count = len(pricingDictionary) + len(indexDictionary) + len(companyDictionary)
    print(str(count) + ' items loaded.')

    # This function is a user request loop.
    getUserRequests()
