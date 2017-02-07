# Transaction class
import sys
import csv
import datatools
import datetime


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


def get_categories(start_date, end_date, transactions):
    categories = dict()
    for transaction in transactions:
        # Do not add any 'Hide' categories.
        if transaction.category.lower()[0:3] == 'hide':
            continue
        if (transaction.transaction_date >= start_date) and (transaction.transaction_date <= end_date):
            if transaction.category in categories:
                categories[transaction.category].append(transaction)
            else:
                categories[transaction.category] = []
                categories[transaction.category].append(transaction)
    return categories


def print_categories(categories):
    for category in categories:
        print_transactions(category, categories[category])


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
        print(pad(transaction.transaction_date, 15) + pad(transaction.description, 30) + pad('${:9,.2f}'.format(transaction.amount),20))
        total += transaction.amount
    print('Total\t' + '${:9,.2f}'.format(total) + '\n')


def print_transaction_totals(title, transactions):
    total = 0
    for transaction in transactions:
        total += transaction.amount
    print(pad(title, 20) + '\t' + '${:9,.2f}'.format(total))


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
    # Main execution
    # Passed arguments
    print(sys.argv)

    transactions = load_transaction_file('transactions.csv')
    start_date = datetime.date(2017, 1, 1)
    end_date = datetime.date(2017, 1, 31)

    credits = get_transactions_by_type(start_date, end_date, transactions, 'credit')
    debits = get_transactions_by_type(start_date, end_date, transactions, 'debit')
    categories = get_categories(start_date, end_date, debits)

    print('\n')
    print(str(start_date) + ' - ' + str(end_date))
    print_transaction_totals('Credits', credits)
    print_transaction_totals('Debits', debits)
    print('\n')
    print_category_totals(categories)
    print_transactions('Credits', debits)
