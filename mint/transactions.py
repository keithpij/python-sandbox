# Transaction class
import sys
import csv
import re
import datetime
import calendar
import datatools
import charting

class Transaction:

    FIELD_COUNT = 0

    DATE_HEADER = 'date'
    DESCRIPTION_HEADER = 'description'
    ORIGINAL_DESCRIPTION_HEADER = 'original description'
    AMOUNT_HEADER = 'amount'
    TRANSACTION_TYPE_HEADER = 'transaction type'
    CATEGORY_HEADER = 'category'
    ACCOUNT_NAME_HEADER = 'account name'
    LABELS_HEADER = 'labels'
    NOTES_HEADER = 'notes'

    index_dict = dict()

    def __init__(self, transaction_details):

        d = Transaction.index_dict
        self.transaction_date = datatools.to_date_slash(transaction_details[d[Transaction.DATE_HEADER]])
        self.description = transaction_details[d[Transaction.DESCRIPTION_HEADER]]
        self.original_description = transaction_details[d[Transaction.ORIGINAL_DESCRIPTION_HEADER]]
        self.amount = float(transaction_details[d[Transaction.AMOUNT_HEADER]])
        self.transaction_type = transaction_details[d[Transaction.TRANSACTION_TYPE_HEADER]]
        self.category = transaction_details[d[Transaction.CATEGORY_HEADER]]
        self.account_name = transaction_details[d[Transaction.ACCOUNT_NAME_HEADER]]
        self.labels = transaction_details[d[Transaction.LABELS_HEADER]]
        self.notes = transaction_details[d[Transaction.NOTES_HEADER]]


def set_indecies(header_list):
    # The field count from the header line will help to correct scenarios
    # where the user put a comma in the Account Name field.
    global FIELD_COUNT
    FIELD_COUNT = len(header_list)

    # By not hard coding these value we are insulated from changes to the
    # underlying export file.
    for i in range(0, FIELD_COUNT):
        Transaction.index_dict[header_list[i].lower()] = i


def load_transaction_file(file):

    # Setup needed variables
    line_count = 0
    transactions = []
    fhand = open(file, 'r')
    reader = csv.reader(fhand)

    # Read through each line of the file.
    for line in reader:
        line_count += 1

        # The first line contains the column headers.
        if line_count == 1:
            set_indecies(line)
        else:
            transaction = Transaction(line)
            # Do not add any transactions in the a 'Hide*' category.
            if transaction.category.lower()[0:4] == 'hide':
                print('Hidden transaction: ' + str(transaction.transaction_date) + ' ' + transaction.description)
            else:
                transactions.append(transaction)

    return transactions


def get_categories(transactions):
    categories = dict()
    for transaction in transactions:
        # Do not add any 'Hide' categories.
        if transaction.category.lower()[0:3] == 'hide':
            continue
        elif transaction.category in categories:
            categories[transaction.category].append(transaction)
        else:
            categories[transaction.category] = []
            categories[transaction.category].append(transaction)
    return categories


def get_category(search_name):
    categories = get_categories(TRANSACTIONS)
    for category_name in categories:
        if re.search(search_name.lower(), category_name.lower()):
        return categories[category_name]
    else:
        return None


def print_categories(categories):
    for category in categories:
        print_transactions(category, categories[category])


def get_category_totals(categories):
    # Create a dictionary of totals for each category.
    category_totals = dict()
    for category_name in categories.keys():
        total = 0
        for transaction in categories[category_name]:
            total += transaction.amount
        category_totals[category_name] = total
    return category_totals


def print_category_totals(categories):
    # Create a dictionary of totals for each category.
    category_totals = dict()
    for category_name in categories.keys():
        total = 0
        for transaction in categories[category_name]:
            total += transaction.amount
        category_totals[category_name] = total

    # Loop through and print the totals for each category.
    for category_name in sorted(category_totals, key=category_totals.__getitem__, reverse=True):  #sorted(d, key=d.__getitem__)
        category_total = category_totals[category_name]
        print(pad(category_name, 20) + '\t' + '${:9,.2f}'.format(category_total))


def get_transactions_by_type(start_date, end_date, transactions, type):
    debits = []
    for transaction in transactions:
        if (transaction.transaction_date >= start_date) and (transaction.transaction_date <= end_date):
            if transaction.transaction_type.lower() == type.lower():
                debits.append(transaction)
    return debits


def print_transactions(title, transactions):
    total = 0
    print('\n\n' + title + '\n')
    transactions.sort(key=lambda x: x.transaction_date)
    for transaction in transactions:
        print(pad(transaction.transaction_date, 15) + pad(transaction.description, 40) + pad(transaction.category, 20) + pad('${:9,.2f}'.format(transaction.amount),20))
        total += transaction.amount


def print_transaction_totals(title, transactions):
    total = 0
    for transaction in transactions:
        if transaction.category.lower() != 'credit card payment':
            total += transaction.amount
    print(pad(title, 20) + '\t' + '${:9,.2f}'.format(total))


def print_category_comparison(previous_month_totals, current_month_totals):
    print(pad('Category', 30) + pad('Previous Month', 20) + pad('Current Month', 20) + pad('Difference', 20))

    for category_name in sorted(previous_month_totals, key=previous_month_totals.__getitem__, reverse=True):
        previous = previous_month_totals[category_name]
        if category_name in current_month_totals:
            current = current_month_totals[category_name]
        else:
            current = 0
        diff = previous - current
        print(pad(category_name, 30) + pad('${:9,.2f}'.format(previous), 20) + pad('${:9,.2f}'.format(current), 20) + pad('${:9,.2f}'.format(diff), 20))


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
    global START_DATE
    global END_DATE
    global TRANSACTIONS

    # User request loop
    while True:
        prompt = str(START_DATE) + ' - ' + str(END_DATE)
        command = input(prompt + ' --> ')

        if command[0:2] == 'dr':
            params = command[2:].strip()
            params_list = params.split(' ')
            start_date = params_list[0]
            end_date = params_list[1]
            START_DATE = datatools.to_date(start_date)
            END_DATE = datatools.to_date(end_date)
            continue

        if command[0:1] == 'c':
            params = command[1:].strip()
            params_list = params.split(' ')
            category = params_list[0]
            if len(category) == 0:
                debits = get_transactions_by_type(START_DATE, END_DATE, TRANSACTIONS, 'debit')
                categories = get_categories(debits)
                print_category_totals(categories)
            else:
                category_transactions = get_category(category)
                print_transactions(category, category_transactions)
            continue

        if command == 'income':
            credits = get_transactions_by_type(START_DATE, END_DATE, TRANSACTIONS, 'credit')
            print_transactions('Credits', credits)
            print_transaction_totals('Credits', credits)
            continue

        if command == 'spending':
            debits = get_transactions_by_type(START_DATE, END_DATE, TRANSACTIONS, 'debit')
            print_transactions('Debits', debits)
            print_transaction_totals('Debits', debits)
            continue

        if command == 'categories':
            debits = get_transactions_by_type(START_DATE, END_DATE, TRANSACTIONS, 'debit')
            categories = get_categories(debits)
            print_category_totals(categories)
            continue

        if command == 'compare':
            now = datetime.datetime.now()
            year = now.year
            month = now.month
            # returns a touple which is the day of the week of the first day of the month and
            # the number of days in the month.
            last_day = calendar.monthrange(year, month)[1]
            start_date = datetime.date(year, month, 1)
            end_date = datetime.date(year, month, last_day)
            debits = get_transactions_by_type(start_date, end_date, TRANSACTIONS, 'debit')
            current_month = get_categories(debits)
            current_month_totals = get_category_totals(current_month)
            #print(str(start_date) + ' ' + str(end_date))
            #print_category_totals(current_month)

            if month == 1:
                year -= 1
                month = 12
            else:
                month -=1

            last_day = calendar.monthrange(year, month)[1]
            start_date = datetime.date(year, month, 1)
            end_date = datetime.date(year, month, last_day)
            debits = get_transactions_by_type(start_date, end_date, TRANSACTIONS, 'debit')
            previous_month = get_categories(debits)
            previous_month_totals = get_category_totals(previous_month)
            #print(str(start_date) + ' ' + str(end_date))
            #print_category_totals(previous_month)
            print_category_comparison(previous_month_totals, current_month_totals)
            continue

        if command == 'help':
            param = command[4:].strip()
            if len(param) > 0:
                print_command_help(param)
            else:
                print_menu()
            continue

        if command == 'lf':
            print('Reading transaction file ...')
            transactions = load_transaction_file('transactions.csv')
            continue

        if command == 'pie':
            debits = get_transactions_by_type(START_DATE, END_DATE, TRANSACTIONS, 'debit')
            categories = get_categories(START_DATE, END_DATE, debits)
            charting.category_pie_chart(categories)
            continue

        if command == 'quit' or command == 'q':
            break

        print('*** Unrecognized command ***')


if __name__ == '__main__':
    # Main execution
    # Passed arguments
    print(sys.argv)

    TRANSACTIONS = load_transaction_file('transactions.csv')
    START_DATE = datetime.date(2017, 1, 1)
    END_DATE = datetime.date(2017, 1, 31)

    # This function is a user request loop.
    get_user_requests()
