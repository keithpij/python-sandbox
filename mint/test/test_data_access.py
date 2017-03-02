import datetime
import calendar
import unittest
import transactions
#from transactions import load_transaction_file

'''
TestDataAccess inherits the following:
- assertTrue
- assertFalse
- assertEqual
- assertRaises
'''

DATA_FILE = 'test_transactions.csv'

class TestDataAccess(unittest.TestCase):


    def setUp(self):
        self.transactions = transactions.load_transaction_file(DATA_FILE)


    def tearDown(self):
        self.transactions = None


    def test_load_transaction_file(self):
        self.assertTrue(self.transactions != None)


    def test_get_income(self):

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
        self.assertTrue(credits != None)


    def test_get_spending(self):

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
        self.assertTrue(debits != None)


if __name__ == '__main__':
    # Method names that represent tests must start with the string 'test'.
    # This naming convention informs the test runner which methods represent tests.
    #unittest.main()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataAccess)
    unittest.TextTestRunner(verbosity=2).run(suite)

    '''
    The following command can be used to run all the tests in this module.
    This command should be run from the 'mint' directory.
    $ python3 -m unittest -v -b test.test-data-access

    Notes:

    The buffer option (-b) will only buffer the output if there are no errors. If an error occures within one
    of the unit tests then all the output will be sent to the console.

    The spirit of unit testing is to start with every individual 'unit' or function within your code.
    A good starting point for designing test suites is to have at a minimum one test for every method/function
    in your code. From there you can take a look at the the data you are sending through your unit test
    to see if it makes sense to add additional unit test that test your code with different data.

    Create multiple unit test for a function when different data causes different branches to execute.
    Be guided by code coverage branche reports.

    '''
