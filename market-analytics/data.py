import time
import datetime
import re
import company
import filetools
import portfolio

COMPANIES = None
COMPANY_PRICING = None
INDEX_PRICING = None
SECTORS = None
INDUSTRIES = None
PORTFOLIOS = None
START_DATE = None
END_DATE = None


def set_date_range_by_days_in_past(daysinpast):
    global START_DATE
    global END_DATE

    END_DATE = datetime.date.today()
    daysInPastDelta = datetime.timedelta(days=daysinpast)
    START_DATE = END_DATE - daysInPastDelta


def load_data():
    '''
    Loads all the data needed by the market analysis application.
    Returns the elapsed time used to load the data.
    '''

    global COMPANIES
    global COMPANY_PRICING
    global INDEX_PRICING
    global SECTORS
    global INDUSTRIES
    global PORTFOLIOS

    start = time.time()
    COMPANIES = company.Companies()
    COMPANY_PRICING = filetools.loadPricingFiles()
    INDEX_PRICING = filetools.loadIndexFiles()
    SECTORS, INDUSTRIES = COMPANIES.getSectorsAndIndustries()
    PORTFOLIOS = portfolio.load_portfolios(COMPANY_PRICING)
    elapsed = time.time() - start

    return elapsed


def get_price_by_date_range(symbol):

    # All the pricing info for the specified ticker.
    company_pricing = COMPANY_PRICING[symbol]

    # Search for date range matches.
    r = dict()
    for date in sorted(company_pricing.keys()):
        if date >= START_DATE and date <= END_DATE:
            r[date] = company_pricing[date]
    return r


def search_companies_by_name(regex):

    matches = dict()
    for symbol in COMPANIES.Dictionary:
        if re.search(regex, COMPANIES.Dictionary[symbol].Name.lower()):
            matches[symbol] = COMPANIES.Dictionary[symbol].Name
    return matches


def find_winners_by_date_range(minprice, maxprice, mingain, maxgain):

    winners = dict()
    for s in COMPANY_PRICING.keys():
        # All the pricing info for the specified ticker.
        pricing = COMPANY_PRICING[s]

        # Search for date range matches.
        r = dict()
        for date in sorted(pricing.keys()):
            if date >= START_DATE and date <= END_DATE:
                r[date] = pricing[date]

        dateList = sorted(r.keys())

        if len(dateList) > 0:
            firstKey = dateList[0]
            lastKey = dateList[-1]
            first = r[firstKey]
            last = r[lastKey]
            g = (last.close - first.open) / first.open
            if first.open > minprice and last.close < maxprice and g > mingain and g < maxgain:
                winners[s] = g

    return winners


def find_winners_by_date(enddate):

    delta = datetime.timedelta(days=1)
    startdate = enddate - delta

    winners = dict()
    for s in COMPANY_PRICING.keys():
        # All the pricing info for the specified ticker.
        pricing = COMPANY_PRICING[s]

        # Search for date range matches.
        r = dict()
        for date in sorted(pricing.keys()):
            if date >= startdate and date <= enddate:
                r[date] = pricing[date]

        dateList = sorted(r.keys())

        if len(dateList) > 0:
            firstKey = dateList[0]
            lastKey = dateList[-1]
            first = r[firstKey]
            last = r[lastKey]
            g = (last.close - first.open) / first.open
            if g > 0:
                winners[s] = g

    return winners


def get_date_range_display():
    return str(START_DATE) + ' - ' + str(END_DATE)
