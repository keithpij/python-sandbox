'''
Market pricing classes and functions.
'''
import os
import glob
import datatools


class MyBase:

    def __init__(self, symbol, date):
        self.Symbol = symbol
        self.Date = date


    def gatherAttributes(self):
        attrs = []
        for key in sorted(self.__dict__):
            attrs.append('%s=%s' % (key, getattr(self,key)))
        return ', '.join(attrs)

    def __repr__(self):
        return '[%s: %s]' % (self.__class__.__name__, self.gatherAttrs())


class Index(MyBase):

    def __init__(self, tickerList):

        symbol = tickerList[0]
        date = datatools.toDate(tickerList[1])
        openprice = float(tickerList[2])
        high = float(tickerList[3])
        low = float(tickerList[4])
        close = float(tickerList[5])
        volume = int(tickerList[6])

        MyBase.__init__(self, symbol, date)
        #self.Symbol = symbol
        #self.Date = date
        self.Open = openprice
        self.High = high
        self.Low = low
        self.Close = close
        self.Change = self.Close - self.Open
        self.Volume = volume


class Price(MyBase):

    # Google BigQuery schema and gcs location.
    # symbol:STRING,date:STRING,open:FLOAT,high:FLOAT,low:FLOAT,close:FLOAT,volume:INTEGER
    # gs://keithpij-bigdata-lab.appspot.com/pricing_2016/*.txt.gz

    def __init__(self, data_list=None):

        if data_list != None:
            symbol = data_list[0]
            date = datatools.toDate(data_list[1])
            openprice = float(data_list[2])
            high = float(data_list[3])
            low = float(data_list[4])
            close = float(data_list[5])
            volume = int(data_list[6])

            MyBase.__init__(self, symbol, date)
            #self.Symbol = symbol
            #self.Date = date
            self.open = openprice
            self.high = high
            self.low = low
            self.close = close
            self.change = self.close - self.open
            self.volume = volume
