# Company Model
import os
import glob
import csv
import sys
import io
import shelve
import datatools
import filetools


COMPANIES_BUCKET_NAME = 'keithpij-company-data'

class Company:

    def __init__(self, date, companyList):

        symbolIndex = 0
        nameIndex = 1
        marketCapIndex = 3
        adrtsoIndex = 4
        ipoYearIndex = 5
        sectorIndex = 6
        industryIndex = 7

        symbol = companyList[symbolIndex]
        name = companyList[nameIndex]
        marketCap = companyList[marketCapIndex]
        adrtso = companyList[adrtsoIndex]
        ipoYear = companyList[ipoYearIndex]
        sector = companyList[sectorIndex]
        industry = companyList[industryIndex]

        self.Symbol = symbol.strip()
        self.Name = name.strip()
        self.Date = date
        self.MarketCap = datatools.toFloat(marketCap)
        self.ADRTSO = adrtso.strip()
        self.IPOYear = datatools.toInt(ipoYear)
        self.Sector = sector.strip()
        self.Industry = industry.strip()
        self.CurrentPrice = 0


class Companies:

    def __init__(self, companyDictionary=None):
        if companyDictionary == None:
            self.Get()
        else:
            self.Dictionary = companyDictionary

        sectors, industries = self.getSectorsAndIndustries()
        self.Sectors = sectors
        self.Industries = industries


    def getSectorsAndIndustries(self):
        # Create the sector and industry dictionaries.
        sectors = dict()
        industries = dict()

        for symbol in self.Dictionary.keys():
            c = self.Dictionary[symbol]

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
                # See if the company's industry is in the industry dictionary.
                if c.Industry not in i:
                    i[c.Industry] = 1
                else:
                    i[c.Industry] = i[c.Industry] + 1
                # Not sure if the line below is neccessary.
                sectors[c.Sector] = d

            # Industry data
            if c.Industry not in industries:
                industries[c.Industry] = 1
            else:
                industries[c.Industry] = industries[c.Industry] + 1

        return sectors, industries


    def Save(self):
        db = shelve.open('db/marketdata')
        db['companies'] = self.Dictionary
        db.close()


    def Get(self):
        db = shelve.open('db/marketdata')
        self.Dictionary = db['companies']
        db.close()


if __name__ == '__main__':

    # Load the NASDAQ and NYSE files that contain company information.
    '''
    companyDictionary = filetools.loadCompanyFiles()
    companies = Companies(companyDictionary)
    companies.Save()
    '''

    companies = Companies()

    print(str(len(companies.Dictionary)) + ' companies.')

    if len(sys.argv) > 1 and sys.argv[1] == '-ls':
        print('\n\nSECTORS\n')
        lineCount = 0
        for s in companies.Sectors.keys():
            lineCount = lineCount + 1
            d = companies.Sectors[s]
            print(str(lineCount) + '. \t' + s + ' ' + str(d['count']))
            for i in d['industries']:
                print('\t\t' + i)


    if len(sys.argv) > 1 and sys.argv[1] == '-li':
        print('\n\nINDUSTRIES\n')
        count = 0
        for i in companies.Industries.keys():
            count = count + 1
            print(str(count) + '. \t' + i + ' ' + str(companies.Industries[i]))
