# Company Models
import os
import glob
import DataTools
import csv
import sys

class Company:

    def __init__(self, date, companyData):

        symbolIndex = 0
        nameIndex = 1
        marketCapIndex = 3
        adrtsoIndex = 4
        ipoYearIndex = 5
        sectorIndex = 6
        industryIndex = 7

        symbol = companyData[symbolIndex]
        name = companyData[nameIndex]
        marketCap = companyData[marketCapIndex]
        adrtso = companyData[adrtsoIndex]
        ipoYear = companyData[ipoYearIndex]
        sector = companyData[sectorIndex]
        industry = companyData[industryIndex]

        self.Symbol = symbol.strip()
        self.Name = name.strip()
        self.MarketCap = DataTools.toFloat(marketCap)
        self.ADRTSO = adrtso.strip()
        self.IPOYear = DataTools.toInt(ipoYear)
        self.Sector = sector.strip()
        self.Industry = industry.strip()
        self.Date = date


def LoadFiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data')
    fileSearch = os.path.join(pricingDir, '*companylist*.csv')

    allFiles = glob.glob(fileSearch)

    companyDictionary = parseFiles(allFiles)

    return companyDictionary


def parseFiles(files):
    companyCount = 0
    companyDictionary = dict()
    for file in files:
        a = file.split('.')
        b = a[0]  # get rid of the file extension
        c = b.split('-')
        date = DataTools.toDate(c[-3] + c[-2] + c[-1])
        print(file)
        print(date)

        # Read through each line of the file.
        lineCount = 0
        fhand = open(file)
        reader = csv.reader(fhand)
        for line in reader:
            lineCount = lineCount + 1

            # The first line contains the column headers.
            if lineCount > 1:
                c = Company(date, line)
                companyDictionary[c.Symbol] = c
                companyCount = companyCount + 1

    return companyDictionary, companyCount


def getUniqueSectorsIndustries(companyDictionary):
    # Create the sector and industry dictionaries.
    sectors = dict()
    industries = dict()

    for symbol in companyDictionary.keys():
        c = companyDictionary[symbol]

        if c.Sector not in sectors:
            sectors[c.Sector] = 1
        else:
            sectors[c.Sector] = sectors[c.Sector] + 1

        if c.Industry not in industries:
            industries[c.Industry] = 1
        else:
            industries[c.Industry] = industries[c.Industry] + 1

    return sectors, industries


if __name__ == '__main__':
    companyDictionary, count = LoadFiles()
    print(str(count) + ' companies.')

    # Get a dictionary of sectors and industries
    sectors, industries = getUniqueSectorsIndustries(companyDictionary)

    if len(sys.argv) > 1 and sys.argv[1] == '-s':
        print('\n\nSECTORS\n')
        count = 0
        for s in sectors.keys():
            count = count + 1
            print(str(count) + '. \t' + s + ' ' + str(sectors[s]))

    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        print('\n\nINDUSTRIES\n')
        count = 0
        for i in industries.keys():
            count = count + 1
            print(str(count) + '. \t' + i + ' ' + str(industries[i]))
