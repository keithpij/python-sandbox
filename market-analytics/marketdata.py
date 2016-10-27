# Market Data
import datetime
import os
import sys
import models
import company
import portfolio
import search
import datatools
import indexutilities
import filetools


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


def listSectors():
    print('\n\nSECTORS\n')
    lineCount = 0
    for s in sectors.keys():
        lineCount = lineCount + 1
        d = sectors[s]
        print(str(lineCount) + '. \t' + s + ' ' + str(d['count']))
        #for i in d['industries']:
        #    print('\t\t' + i)


def listIndustries():
    print('\n\nINDUSTRIES\n')
    lineCount = 0
    for i in industries.keys():
        lineCount += 1
        d = industries[i]
        print(str(lineCount) + '. \t' + i + ' ' + str(industries[i]))


def showHelp():
    print('-q to quit')
    print('-ls to list all sectors')
    print('-li to list all industries')
    print('-h to show this help menu')
    print('-w to find winners')
    print('-d to change the days in the past for which pricing data is loaded')
    print('-s to search the pricing dictionary for a ticker')
    print('-p to print the latest pricing data for the portfolio')


def getUserRequests():
    # global variable
    global searchcriteria

    # User request loop
    while True:
        command = input(str(searchcriteria) + ' --> ')

        # Quit the input loop.
        if command == '-q':
            break

        if command == '-ls':
            listSectors()

        if command == '-li':
            listIndustries()

        if command == '-h':
            showHelp()

        if command == '-w':
            winners = findWinners()
            printWinners(winners)

        if command == '-zip':
            filetools.zipTextFiles()

        # Change days in the past.
        if command[0:2] == '-d':
            days = command[3:]
            d = DataTools.toInt(days)
            searchcriteria = search.Criteria(daysinpast=d)

        # Search by name.  The search text must be a regular expression.
        if command[0:2] == '-s':
            s = command[3:].upper()

            # Check and see if ticker symbol exists.
            if s not in pricingDictionary:
                print('Company ' + s + ' not found.')
                continue

            searchcriteria.symbol = s
            searchobj = search.Search(pricingDictionary)
            resultDict = searchobj.getpricebydaterange(searchcriteria)
            printTickerData(s, resultDict)

        if command[0:2] == '-p':
            portfolio.printPortfolio(companyDictionary, indexDictionary, pricingDictionary)


if __name__ == '__main__':
    # Main execution
    # Passed arguments
    print(sys.argv)

    # Start off with a date range that is 30 days in the past.
    global searchcriteria
    searchcriteria = search.Criteria(daysinpast=60)

    # Load company files, company pricing files and index files.
    # Get a dictionary of sectors and industries
    print('Loading Data ...')
    companyDictionary, count = filetools.loadCompanyFiles()
    pricingDictionary = filetools.loadPricingFiles()
    indexDictionary = filetools.loadIndexFiles()
    sectors, industries = company.getSectorsAndIndustries(companyDictionary)
    print(str(count) + ' companies')

    '''
    companyDictionary, count = company.loaddatafromblobs()
    pricingDictionary = models.loaddatafromblobs()
    indexDictionary = indexutilities.loaddatafromblobs()
    '''

    count = len(pricingDictionary) + len(indexDictionary) + len(companyDictionary)
    print(str(count) + ' items loaded.')

    # This function is a user request loop.
    getUserRequests()
