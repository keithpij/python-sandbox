'''
Module to keep track of holdings within a portfolio
'''
import datatools
import settings
import filetools


def load_portfolios(pricing_dictionary):
    '''
    Parameters: pricing_dictionary - a dictionary of symbols containing
    princing information.
    Returns a list of portfolios.
    '''
    portfolios = []
    for file in settings.Portfolios:
        holdings_dict = dict()
        fhand = open(file)
        linecount = 0
        for line in fhand:
            linecount += 1
            if linecount == 1:
                name = line.strip()
            else:
                data_list = line.split(',')
                symbol = data_list[0]
                shares = datatools.toInt(data_list[1])
                purchase_date = datatools.toDate(data_list[2])
                holding_obj = Holding(symbol, shares, purchase_date, pricing_dictionary)
                holdings_dict[symbol] = holding_obj

        port = Portfolio(name, holdings_dict)
        portfolios.append(port)
    return portfolios


def get_price_by_date(pricing_dictionary, symbol, pricingdate):
    '''
    Return the price for a single company on a specified date.
    '''
    # Symbol not found
    if symbol.upper() not in pricing_dictionary:
        return 0

    # Date match
    days_dict = pricing_dictionary[symbol.upper()]
    if pricingdate in days_dict:
        return days_dict[pricingdate].close

    # Look for first date that is greater than pricing date.
    days_list_sorted = sorted(days_dict)
    for i in days_list_sorted:
        if days_dict[i].Date > pricingdate:
            return days_dict[i].close

    # If you get here then return the most recent price.
    last_day = days_list_sorted[-1]
    return days_dict[last_day].close


def get_current_price(pricing_dictionary, symbol):
    '''
    Retrieves the latest price for a specified company (symbol).
    '''
    if symbol.upper() not in pricing_dictionary:
        return 0

    days_dict = pricing_dictionary[symbol.upper()]
    days_list_sorted = sorted(days_dict)
    last_day = days_list_sorted[-1]
    price = days_dict[last_day]
    return price.close


class Portfolio:
    '''
    Class to hold portfolio information and methods.
    '''
    def __init__(self, name, holdings):
        self.name = name
        self.holdings = holdings
        self.value = self.get_portfolio_value()
        self.gain = self.get_gain()


    def get_portfolio_value(self):
        '''
        Calculate and return the total value of the portfolio.
        '''
        total_value = 0.0
        for symbol in self.holdings.keys():
            holding_value = self.holdings[symbol].current_price * self.holdings[symbol].shares
            total_value += holding_value
        return total_value


    def get_gain(self):
        '''
        Calculate and return the total gain for this portfolio.
        '''
        portfolio_gain = 0.0
        for symbol in self.holdings.keys():
            portfolio_gain += self.holdings[symbol].total_gain
        return portfolio_gain


class Holding:
    '''
    Class used to encapsulate a portfolio holding.
    '''

    def __init__(self, symbol, shares, purchase_date, pricing_dictionary):
        self.symbol = symbol
        self.shares = shares
        self.purchase_date = purchase_date
        self.purchase_price = get_price_by_date(pricing_dictionary, symbol, purchase_date)
        self.current_price = get_current_price(pricing_dictionary, symbol)
        self.gain = self.current_price - self.purchase_price
        self.total_gain = self.shares * self.gain


def pad(value, width):
    '''
    Turns value into a string and then pads the new string value with spaces
    to produce a string of length width.
    '''
    display = str(value)
    length = len(display)

    if length >= width:
        return display[0:width]

    delta = width - length
    for _ in range(delta):
        display = display + ' '
    return display


if __name__ == '__main__':

    PRICING_DICTIONARY = filetools.loadPricingFiles()
    PORTFOLIO_LIST = load_portfolios(PRICING_DICTIONARY)
    report()


def report():
    '''
    Print a holdings report and a portfolio value report.
    '''
    for portfolio in PORTFOLIO_LIST:
        for symbol in portfolio.Holdings.keys():
            holding = portfolio.Holdings[symbol]
            print(holding.Symbol + ' ' + str(holding.Shares) + ' ' +
                  str(holding.PurchaseDate) + ' ' +
                  str(holding.PurchasePrice) + ' ' +
                  str(holding.CurrentPrice))

    print(pad('Portfolio Name', 40) + pad('Current Value', 15) + pad('Gain', 15))
    for port in PORTFOLIO_LIST:
        print(pad(port.Name, 40) +
              pad('${:7,.2f}'.format(float(port.Value)), 15) +
              pad('${:7,.2f}'.format(float(port.Gain)), 15))
