# Company Model
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
        self.CurrentPrice = 0


def getCompanies():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = os.path.join('/Users', 'keithpij', 'Documents')
    #DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data')
    fileSearch = os.path.join(pricingDir, '*companylist*.csv')

    allFiles = glob.glob(fileSearch)

    companyDictionary, count = parseFiles(allFiles)

    return companyDictionary, count


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


def getSectorsAndIndustries(companyDictionary):
    # Create the sector and industry dictionaries.
    sectors = dict()
    industries = dict()

    for symbol in companyDictionary.keys():
        c = companyDictionary[symbol]

        # Sector data
        if c.Sector not in sectors:
            d = dict()
            d['count'] = 1
            i = dict()
            i[c.Industry] = 1
            d['industries'] = i
            sectors[c.Sector] = d

        else:
            d = sectors[c.Sector]
            d['count'] = d['count'] + 1
            i = d['industries']
            if c.Industry not in i:
                i[c.Industry] = 1
            else:
                i[c.Industry] = i[c.Industry] + 1
            sectors[c.Sector] = d

        # Industry data
        if c.Industry not in industries:
            industries[c.Industry] = 1
        else:
            industries[c.Industry] = industries[c.Industry] + 1

    return sectors, industries


if __name__ == '__main__':

    # Load the NASDAQ and NYSE files that contain company information.
    companyDictionary, count = getCompanies()
    print(str(count) + ' companies.')

    # Get a dictionary of sectors and industries
    sectors, industries = getSectorsAndIndustries(companyDictionary)

    if len(sys.argv) > 1 and sys.argv[1] == '-s':
        print('\n\nSECTORS\n')
        lineCount = 0
        for s in sectors.keys():
            lineCount = lineCount + 1
            d = sectors[s]
            print(str(lineCount) + '. \t' + s + ' ' + str(d['count']))
            for i in d['industries']:
                print('\t\t' + i)


    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        print('\n\nINDUSTRIES\n')
        count = 0
        for i in industries.keys():
            count = count + 1
            print(str(count) + '. \t' + i + ' ' + str(industries[i]))
