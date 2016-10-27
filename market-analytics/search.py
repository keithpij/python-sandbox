'''
This class is used to hold search criteria.
'''

import datetime

class Criteria:


    def __init__(self, startdate=None, enddate=None, daysinpast=None, sector=None, industry=None, symbol=None, pricerangelow=None, pricerangehigh=None, gainlow=None, gainhigh=None):

        self.startdate = startdate
        self.enddate = enddate
        self.daysinpast = daysinpast
        self.sector = sector
        self.industry = industry
        self.symbol = symbol
        self.pricerangelow = pricerangelow
        self.pricerangehigh = pricerangehigh
        self.gainlow = gainlow
        self.gainhigh = gainhigh

        # Passing in a days in the past parameter will override start date and end date.
        if not daysinpast == None:
            self.enddate = datetime.date.today()
            daysInPastDelta = datetime.timedelta(days=daysinpast)
            self.startdate = self.enddate - daysInPastDelta


    def __repr__(self):
        return str(self.startdate) + ' - ' + str(self.enddate)


class Search:

    def __init__(self, pricingDictionary):
        self.pricingDictionary = pricingDictionary
        self.criteria = None


    def getpricebydaterange(self, criteria):
        # Save the criteria.
        self.criteria = criteria

        # All the pricing info for the specified ticker.
        companypricingdict = self.pricingDictionary[criteria.symbol]

        # Search for date range matches.
        r = dict()
        for date in sorted(companypricingdict.keys()):
            if date >= self.criteria.startdate and date <= self.criteria.enddate:
                r[date] = companypricingdict[date]
        return r
