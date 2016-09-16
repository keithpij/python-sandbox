'''
Module to keep track of holdings within a portfolio
'''
import csv
import sys
import os
import datetime
import DataTools
import company
import price
import index


class Portfolio:

    def __init__(self, holdings):
        self.Holdings = holdings
        self.PortfolioValue = Portfolio.getportfoliovalue(holdings)


    def getportfoliovalue(holdings):
        v = 0.0
        for symbol in holdings.keys():
            v = v + holdings[symbol].company.CurrentPrice
        return v


    def loadportfolio(companyDictionary):
        # currentWorkingDir = os.getcwd()
        DATA_DIR = '/Users/keithpij/Documents'
        pricingDir = os.path.join(DATA_DIR, 'eod-data')
        portfolioFile = os.path.join(pricingDir, 'portfolio.txt')
        portfolio = dict()
        fhand = open(portfolioFile)
        for symbol in fhand:
            symbol = symbol.strip()
            company = companyDictionary[symbol]
            h = Holding(company, 100, '20160404')
            portfolio[symbol] = h

        return portfolio


class Holding:

    def __init__(self, company, shares, purchasedate):
        self.company = company
        self.shares = DataTools.toInt(shares)
        self.purchasedate = DataTools.toDate(purchasedate)


def printPortfolio(companyDictionary, indexDictionary, pricingDictionary):
    today = datetime.date.today()
    print('\nToday\'s date: ' + str(today) + '\n')

    # Print the last pricing information for each ticker in the portfolio
    print('Ticker\tDate\t\tOpen\t\tHigh\t\tLow\t\tClose\t\tChange\t\tVolume')
    # Print the indecies.
    index.PrintLastDateForIndex(indexDictionary, 'DJI')
    index.PrintLastDateForIndex(indexDictionary, 'NAST')

    # Load the portfolio file.
    holdings = Portfolio.loadportfolio(companyDictionary)  # Dictionary of Holdings.
    portfolio = Portfolio(holdings)

    # Loop through the portfolio.
    for symbol in portfolio.Holdings:
        printLastDateForTicker(pricingDictionary, symbol)


def printLastDateForTicker(pricingDictionary, symbol):
    if symbol.upper() not in pricingDictionary:
        return

    daysDict = pricingDictionary[symbol.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    p = daysDict[lastDay]

    # create the display string
    d = p.Symbol + '\t' + str(p.Date) + '\t'
    d = d + '${:7,.2f}'.format(float(p.Open)) + '\t'
    d = d + '${:7,.2f}'.format(float(p.High)) + '\t'
    d = d + '${:7,.2f}'.format(float(p.Low)) + '\t'
    d = d + '${:7,.2f}'.format(float(p.Close)) + '\t'
    d = d + '${:7,.2f}'.format(p.Change) + '\t'
    d = d + '{:11,.0f}'.format(int(p.Volume))
    print(d)


if __name__ == '__main__':

    # Load company, pricing, and index files.
    print('Loading Data ...')
    companyDictionary, count = company.getallcompaniesfromfiles()
    pricingDictionary = price.LoadFiles()
    indexDictionary = index.LoadFiles()

    # Print the portfolio.
    printPortfolio(companyDictionary, indexDictionary, pricingDictionary)
