import datetime
import calendar
import transactions
#from transactions import load_transaction_file


DATA_FILE = 'transactions.csv'


def test_load_transaction_file():
    trans = transactions.load_transaction_file(DATA_FILE)
    assert trans != None


def test_get_income():
    trans = transactions.load_transaction_file(DATA_FILE)

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
    credits = transactions.get_transactions_by_type(start_date, end_date, trans, 'credit')
    assert credits != None


def test_get_spending():
    trans = transactions.load_transaction_file(DATA_FILE)

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
    debits = transactions.get_transactions_by_type(start_date, end_date, trans, 'debit')
    assert debits != None


if __name__ == '__main__':
    # Method names that represent tests must start with the string 'test'.
    # This naming convention informs the test runner which methods represent tests.
    #unittest.main()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataAccess)
    unittest.TextTestRunner(verbosity=2).run(suite)
