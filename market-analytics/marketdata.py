'''
Main entry module for the Market Data application.
'''
import sys
import re
import datatools
import filetools
import webtools
import data


def get_pricing_history(symbol):
    '''
    Will return a dictionary of pricing information for the specified symbol.
    Uses the SEARCH_CRITERIA object.
    '''
    symbol = symbol.upper()

    # Check and see if ticker symbol exists.
    if symbol not in data.COMPANY_PRICING:
        print('Company ' + symbol + ' not found.')
    else:
        result_dict = data.get_price_by_date_range(symbol)
        print_pricing_details(symbol, result_dict)


def print_pricing_details(symbol, daily_pricing_dict):
    '''
    Prints pricing and summary information for the symbol passed in.
    Also needs the daily pricing dictionary for the symbol to be passed in.
    '''
    if symbol in data.COMPANIES.Dictionary.keys():
        company_name = data.COMPANIES.Dictionary[symbol].Name
    else:
        company_name = symbol

    print('\n' + company_name + ' (' + symbol + ')\n')
    print(pad('Date', 15) + pad('Open', 15) + pad('Close', 15) +
          pad('Change', 15) + pad('Volume', 0))

    date_list = sorted(daily_pricing_dict.keys())
    for date in date_list:
        price = daily_pricing_dict[date]
        display = pad(str(date), 15)
        display += pad('${:7,.2f}'.format(price.open), 15)
        display += pad('${:7,.2f}'.format(price.close), 15)
        display += pad('${:7,.2f}'.format(price.change), 15)
        display += pad('{:11,.0f}'.format(price.volume), 0)
        print(display)

    start_key = date_list[0]
    end_key = date_list[-1]
    start_price = daily_pricing_dict[start_key].open
    end_price = daily_pricing_dict[end_key].close
    gain = end_price - start_price
    percent_gain = (gain/start_price) * 100
    print('\n')
    print(pad('Start Pricing:', 15) + pad('${:7,.2f}'.format(start_price), 15))
    print(pad('End Pricing:', 15) + pad('${:7,.2f}'.format(end_price), 15))
    print(pad('Gain:', 15) + pad('${:7,.2f}'.format(gain), 15))
    print(pad('Percent Gain:', 15) + pad('%{:7,.1f}'.format(percent_gain), 15))
    print('\n')


def show_portfolio_by_date_range(regex):
    '''
    List the details for a portfolio whose name matches the passed in
    regular expression.
    '''
    for port in data.PORTFOLIOS:
        if re.search(regex.lower(), port.name.lower()):
            print('\n')
            print(port.name)
            print('\n')

            print(pad('Symbol', 10) + pad('Name', 20) +
                  pad('Start Price', 15) + pad('End Price', 15) +
                  pad('Gain', 15) + pad('Percent Gain', 15))

            holdings = port.holdings
            for symbol in holdings:
                # Get the pricing history for the current date range.
                daily_pricing_dict = data.get_price_by_date_range(symbol)

                # Get the full company name
                if symbol in data.COMPANIES.Dictionary.keys():
                    company_name = data.COMPANIES.Dictionary[symbol].Name
                else:
                    company_name = symbol

                date_list = sorted(daily_pricing_dict.keys())

                # Calculations
                start_key = date_list[0]
                end_key = date_list[-1]
                start_price = daily_pricing_dict[start_key].open
                end_price = daily_pricing_dict[end_key].close
                gain = end_price - start_price
                percent_gain = (gain/start_price) * 100

                print(pad(symbol, 10) + pad(company_name, 20) +
                      pad('${:7,.2f}'.format(start_price), 15) +
                      pad('${:7,.2f}'.format(end_price), 15) +
                      pad('${:7,.2f}'.format(gain), 15) +
                      pad('%{:7,.1f}'.format(percent_gain), 15))


def get_pricing_summary(symbol, daily_pricing_dict):
    '''
    Prints summary information for the symbol passed in.
    Also needs the daily pricing dictionary for the symbol passed in.
    '''
    if symbol in data.COMPANIES.Dictionary.keys():
        company_name = symbol + ' ' + data.COMPANIES.Dictionary[symbol].Name
    else:
        company_name = symbol

    date_list = sorted(daily_pricing_dict.keys())

    start_key = date_list[0]
    end_key = date_list[-1]
    start_price = daily_pricing_dict[start_key].Open
    end_price = daily_pricing_dict[end_key].Close
    gain = end_price - start_price
    percent_gain = (gain/start_price) * 100

    print(pad(company_name, 25) +
          pad('${:7,.2f}'.format(start_price), 15) +
          pad('${:7,.2f}'.format(end_price), 15) +
          pad('${:7,.2f}'.format(gain), 15) +
          pad('%{:7,.1f}'.format(percent_gain), 15))


def get_winners_by_date(params):
    '''
    Prints a list of companies that meet the criteria specified by params.
    Params must be a space seperated string containing minprice, maxprice,
    mingain, and maxgain.
    if params is an empty string then the existing criteria will be used.
    '''
    params = params.strip()
    params_list = params.split(' ')
    enddate = datatools.toDate(params_list[0])
    show = datatools.toInt(params_list[1])

    winners = data.find_winners_by_date(enddate)

    # Print the winners
    count = 0
    sorted_winners = sorted(winners, key=winners.__getitem__)

    for symbol in sorted_winners[::-1]:  #winners.keys():
        count = count + 1

        if symbol in data.COMPANIES.Dictionary.keys():
            company_name = data.COMPANIES.Dictionary[symbol].Name
        else:
            company_name = symbol

        print(pad(company_name, 50) + pad(symbol, 10) + '%{:3,.1f}'.format(winners[symbol]*100))

        if count == show:
            break

    print(str(count) + ' companies.')


def get_winners_by_date_range(params):
    '''
    Prints a list of companies that meet the criteria specified by params.
    Params must be a space seperated string containing minprice, maxprice,
    mingain, and maxgain.
    if params is an empty string then the existing criteria will be used.
    '''
    params = params.strip()
    criteria = params.split(' ')
    if len(criteria[0]) > 0:
        minprice = datatools.toFloat(criteria[0])
        maxprice = datatools.toFloat(criteria[1])
        mingain = datatools.toFloat(criteria[2])
        maxgain = datatools.toFloat(criteria[3])

    winners = data.find_winners_by_date_range(minprice, maxprice, mingain, maxgain)

    # Print the winners
    count = 0
    sorted_winners = sorted(winners, key=winners.__getitem__)

    for symbol in sorted_winners:  #winners.keys():
        count = count + 1

        if symbol in data.COMPANIES.Dictionary.keys():
            company_name = data.COMPANIES.Dictionary[symbol].Name
        else:
            company_name = symbol

        print(pad(company_name, 50) + pad(symbol, 10) + '%{:3,.1f}'.format(winners[symbol]*100))

    print(str(count) + ' companies.')


def company_search(regex):
    '''
    Searches the companies dictionary for company names that matches
    the passed in regular expression.
    '''
    matches = data.search_companies_by_name(regex)
    for symbol in matches:
        print(pad(symbol, 10) + pad(matches[symbol], 45))


def list_sectors():
    '''
    Prints all sectors.
    '''
    print('\n\nSECTORS\n')
    count = 0
    for name in data.SECTORS:
        count += 1
        sector_dict = data.SECTORS[name]
        print(str(count) + '. \t' + name + ' ' + str(sector_dict['count']))


def list_industries():
    '''
    Prints all industries.
    '''
    print('\n\nINDUSTRIES\n')
    count = 0
    for i in data.INDUSTRIES:
        count += 1
        print(str(count) + '. \t' + i + ' ' + str(data.INDUSTRIES[i]))


def list_portfolios():
    '''
    Lists all portfolios and their gain.
    '''
    print(pad('Portfolio Name', 40) + pad('Current Value', 15) + pad('Gain', 15))
    for port in data.PORTFOLIOS:
        print(pad(port.name, 40) +
              pad('${:7,.2f}'.format(float(port.value)), 15) +
              pad('${:7,.2f}'.format(float(port.gain)), 15))


def show_portfolio_holdings(regex):
    '''
    List the details for a portfolio whose name matches the passed in
    regular expression.
    '''
    for port in data.PORTFOLIOS:
        if re.search(regex.lower(), port.name.lower()):
            print('\n')
            print(port.name)
            print('\n')

            print(pad('Symbol', 10) + pad('Shares', 10) + pad('Purchase Date', 15) +
                  pad('Purchase Price', 15) + pad('Current Price', 15) +
                  pad('Gain', 15) + pad('Total Gain', 15))

            holdings = port.holdings
            for symbol in holdings:
                print(pad(symbol, 10) + pad(holdings[symbol].shares, 10) +
                      pad(holdings[symbol].purchase_date, 15) +
                      pad('${:7,.2f}'.format(float(holdings[symbol].purchase_price)), 15) +
                      pad('${:7,.2f}'.format(float(holdings[symbol].current_price)), 15) +
                      pad('${:7,.2f}'.format(float(holdings[symbol].gain)), 15) +
                      pad('${:10,.2f}'.format(float(holdings[symbol].total_gain)), 15))
            print('\n')


def show_help():
    '''
    Prints a list of all commands and parameters.
    '''
    print('\n')
    print('-cs [regex] to get a list of all companies whose name matches the regular expression')
    print('-d [days] to change the days in the past for which pricing data is loaded')
    print('-h to show this help menu')
    print('-li to list all industries')
    print('-lp to list the current value and gain of all portfolios')
    print('-ls to list all sectors')
    print('-lph [regex] to list portfolio holdings performance for a portfolio whose name matches regex')
    print('-lpdr [regex] to list portfolio performance by the current date range for a portfolio whose name matches regex')
    print('-price [symbol] to get pricing history for the specified days in the past')
    print('-q to quit')
    print('-wd [date] [number to return]to find winners for a specific date')
    print('-wdr [minprice] [maxprice] [min percent gain] [max percent gain] to find winners by date range')
    print('-zip to compress pricing text files')
    print('\n')


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


def get_user_requests():
    '''
    User request loop. Reads the next command from the user and then calls
    the appropriate function.
    '''

    # User request loop
    while True:
        prompt = data.get_date_range_display()
        command = input(prompt + ' --> ')

        if command[0:3] == '-cs':
            regex = command[4:]
            company_search(regex)

        if command[0:2] == '-d':
            param = command[3:]
            days = datatools.toInt(param)
            data.set_date_range_by_days_in_past(days)

        if command == '-h':
            show_help()

        if command == '-li':
            list_industries()

        if command[0:5] == '-live':
            webtools.get_pricing(command[6:])

        if command == '-lp':
            list_portfolios()

        if command == '-ls':
            list_sectors()

        if command[0:4] == '-lph':
            show_portfolio_holdings(command[5:])

        if command[0:5] == '-lpdr':
            show_portfolio_by_date_range(command[6:])

        if command[0:6] == '-price':
            get_pricing_history(command[7:])

        if command == '-q':
            break

        if command[0:4] == '-wdr':
            get_winners_by_date_range(command[5:])
            continue

        if command[0:3] == '-wd':
            get_winners_by_date(command[4:])

        if command == '-zip':
            count = filetools.zipTextFiles()
            print(str(count) + ' files compressed')


if __name__ == '__main__':
    # Main execution
    # Passed arguments
    print(sys.argv)

    # Start off with a date range that is 60 days in the past.
    data.set_date_range_by_days_in_past(60)

    # Load company files, company pricing files and index files.
    # Get a dictionary of sectors and industries
    print('Loading Data ...')
    ELAPSED = data.load_data()
    print(str(len(data.COMPANIES.Dictionary)) + ' companies')
    print(ELAPSED)

    # This function is a user request loop.
    get_user_requests()
