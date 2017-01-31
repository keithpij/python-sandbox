# Transaction class
import sys
import csv


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
        self.transaction_date = transaction_details[d[Transaction.DATE_HEADER]]
        self.description = transaction_details[d[Transaction.DESCRIPTION_HEADER]]
        self.original_description = transaction_details[d[Transaction.ORIGINAL_DESCRIPTION_HEADER]]
        self.amount = transaction_details[d[Transaction.AMOUNT_HEADER]]
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
            transactions.append(transaction)

    return transactions


if __name__ == '__main__':
    # Main execution
    # Passed arguments
    print(sys.argv)

    transactions = load_transaction_file('transactions.csv')
    for transaction in transactions:
        print(transaction.transaction_date + ' ' + transaction.amount)
