'''
This class is used to hold search criteria.
'''


class Criteria:


    def __init__(self, startdate, enddate, sector=None, industry=None, symbol=None, pricerangelow=None, pricerangehigh=None, gainlow=None, gainhigh=None):

        self.startdate = startdate
        self.enddate = enddate
        self.sector = sector
        self.industry = industry
        self.symbol = symbol
        self.pricerangelow = pricerangelow
        self.pricerangehigh = pricerangehigh
        self.gainlow = gainlow
        self.gainhigh = gainhigh
