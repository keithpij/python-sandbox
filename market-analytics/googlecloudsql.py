'''
Google Cloud SQL utilities
'''
# import MySQLdb
import pymysql.cursors
import company

DROP_COMPANIES_TABLE = 'DROP TABLE IF EXISTS companies'
CREATE_COMPANIES_TABLE = 'CREATE TABLE companies (symbol VARCHAR(10), name TEXT, marketcap REAL, adrtso TEXT, ipoyear INTEGER, sector TEXT, industry TEXT, PRIMARY KEY(symbol) )'
INSERT_COMPANY = 'INSERT INTO companies (symbol, name, marketcap, adrtso, ipoyear, sector, industry) VALUES (%s,%s,%s,%s,%s,%s,%s)'
GET_COMPANY = 'SELECT * FROM companies WHERE symbol = %s'


def getconnection(local):
    if local:
        cn = pymysql.connect(host='localhost', db='marketdata', user='root', password='Local3643')
    else:
        cn = pymysql.connect(host='104.196.135.130', db='marketdata', user='root', password='Cloud')

    return cn


def createcompanytable():
    cn = getconnection(local=True)
    print(cn)
    cur = cn.cursor()
    cur.execute(DROP_COMPANIES_TABLE)
    cur.execute(CREATE_COMPANIES_TABLE)
    cn.commit()
    cn.close()


def insertallcompanies():
    # Get the companies.
    companiesDictionary, totalcompanycount = company.getallcompaniesfromfiles()

    cn = getconnection(local=True)
    cur = cn.cursor()

    insertcount = 0
    for symbol in companiesDictionary.keys():
        c = companiesDictionary[symbol]
        values = c.Symbol, c.Name, c.MarketCap, c.ADRTSO, c.IPOYear, c.Sector, c.Industry
        cur.execute(INSERT_COMPANY, values )
        cn.commit()

        insertcount = insertcount + 1
        if insertcount % 100 == 0:
            print(str(insertcount) + ' inserted of ' + str(totalcompanycount))
    cn.close()


def getcompany(symbol):
    cn = getconnection(local=True)
    cursor = cn.cursor()
    values = (symbol,)
    cursor.execute(GET_COMPANY, values)
    result = cursor.fetchone()
    print(result)
    cn.close()


if __name__ == '__main__':

    # Drop and create a Companies table then insert all companies into it.
    createcompanytable()
    insertallcompanies()

    getcompany('MSFT')
