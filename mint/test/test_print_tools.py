import datetime
import calendar
import unittest
import transactions
import printtools


DATA_FILE = 'test_transactions.csv'

class TestPrintTools(unittest.TestCase):


    def setUp(self):
        self.transactions = transactions.load_transaction_file(DATA_FILE)


    def tearDown(self):
        self.transactions = None


    def test_print_income(self):

        # Get the current year and month.
        now = datetime.datetime.now()
        year = now.year
        month = now.month

        # Get previous month to make sure we have data.
        if month == 1:
            year -= 1
            month = 12
        else:
            month -=1

        # Calculate the last day of the month.
        # calendar.monthrange returns a touple which is the day of the week of the first day of the month and
        # the number of days in the month.
        last_day = calendar.monthrange(year, month)[1]
        start_date = datetime.date(year, month, 1)
        end_date = datetime.date(year, month, last_day)
        credits = transactions.get_transactions_by_type(start_date, end_date, self.transactions, 'credit')
        printtools.print_transactions('Credits', credits)
        printtools.print_transaction_totals('Credits', credits)
        self.assertTrue(credits != None)


    def test_print_credits(self):

        # Get the current year and month.
        now = datetime.datetime.now()
        year = now.year
        month = now.month

        # Get previous month to make sure we have data.
        if month == 1:
            year -= 1
            month = 12
        else:
            month -=1

        # Calculate the last day of the month.
        # calendar.monthrange returns a touple which is the day of the week of the first day of the month and
        # the number of days in the month.
        last_day = calendar.monthrange(year, month)[1]
        start_date = datetime.date(year, month, 1)
        end_date = datetime.date(year, month, last_day)
        debits = transactions.get_transactions_by_type(start_date, end_date, self.transactions, 'debit')
        printtools.print_transactions('Debits', debits)
        printtools.print_transaction_totals('Debits', debits)
        self.assertTrue(debits != None)
