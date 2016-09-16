'''
SQLite utilities
'''
import sqlite3
import company

DROP_COMPANIES_TABLE = 'DROP TABLE IF EXISTS Companies'
CREATE_COMPANIES_TABLE = 'CREATE TABLE Companies (symbol TEXT, name TEXT, marketcap REAL, adrtso TEXT, ipoyear INTEGER, sector TEXT, industry TEXT )'
INSERT_COMPANY = 'INSERT INTO Companies (symbol, name, marketcap, adrtso, ipoyear, sector, industry) VALUES (?,?,?,?,?,?,?)'

def createCompanyTable():
    cn = sqlite3.connect('market.sqlite3')
    cur = cn.cursor()
    cur.execute(DROP_COMPANIES_TABLE)
    cur.execute(CREATE_COMPANIES_TABLE)

    # Get the companies.
    companiesDictionary, count = company.getCompanies()

    for symbol in companiesDictionary.keys():
        c = companiesDictionary[symbol]
        print(symbol)
        values = c.Symbol, c.Name, c.MarketCap, c.ADRTSO, c.IPOYear, c.Sector, c.Industry
        cur.execute(INSERT_COMPANY, values )
        cn.commit()

    cn.close()


if __name__ == '__main__':

    # Load the NASDAQ and NYSE files that contain company information.
    createCompanyTable()
