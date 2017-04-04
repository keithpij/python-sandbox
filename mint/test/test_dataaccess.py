'''
Module to test the dataaccess module.
'''
import datetime
import calendar
import pytest
import settings
import dataaccess


@pytest.fixture(scope='module')
def transactions():
    ''' Fixture that will return a list of transactions.'''
    print('Calling the transaction fixture.')
    trans = dataaccess.load_transaction_file(settings.DATA_FILE)
    yield trans
    trans = None


@pytest.fixture(scope='module')
def previous_month():
    '''
    Fixture that will return a tuple with two date values.
    The first value is the first day of the previous month.
    The second value is the last day of the previous month.
    '''
    # Get the current year and month.
    now = datetime.datetime.now()
    year = now.year
    month = now.month

    # Get previous month to make sure we have data.
    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1

    # Calculate the last day of the month.
    # calendar.monthrange returns a touple which is the day of the week of the first day of the
    # month and the number of days in the month.
    last_day = calendar.monthrange(year, month)[1]
    start_date = datetime.date(year, month, 1)
    end_date = datetime.date(year, month, last_day)
    return (start_date, end_date)


def test_load_transaction_file_1():
    ''' Make sure we have data.'''
    transactions_list = dataaccess.load_transaction_file(settings.DATA_FILE)
    assert transactions_list != None


def test_load_transaction_file_2():
    ''' Make sure the correct exception is raised. '''
    with pytest.raises(dataaccess.MissingTransactionFile):
        transactions = dataaccess.load_transaction_file('does_not_exist.csv')


def test_load_transaction_file_3():
    ''' Make sure the correct message is created with the exception. '''
    with pytest.raises(dataaccess.MissingTransactionFile) as exception_info:
        file_name = 'does_not_exist.csv'
        transactions = dataaccess.load_transaction_file(file_name)
    assert file_name in str(exception_info.value)


def test_get_categories(transactions):
    categories = dataaccess.get_categories(transactions)
    assert categories != None


def test_get_category_by_name_1(transactions):
    search_name = 'mort'
    category = dataaccess.get_category_by_name(search_name, transactions)
    assert category != None


def test_get_category_by_name_2(transactions):
    search_name = 'zzzzz'
    category = dataaccess.get_category_by_name(search_name, transactions)
    assert category == None


def test_get_category_totals(transactions):
    categories = dataaccess.get_categories(transactions)
    category_totals = dataaccess.get_category_totals(categories)
    assert category_totals != None


def test_get_accounts(transactions):
    accounts = dataaccess.get_accounts(transactions)
    assert accounts != None


def test_get_transactions_by_type_1(transactions, previous_month):
    '''
    Will test the get_transactions_by_type function by requesting transactions of type credit.
    '''
    start_date = previous_month[0]
    end_date = previous_month[1]

    credits_list = dataaccess.get_transactions_by_type(start_date, end_date, transactions, 'credit')
    assert credits_list != None


def test_get_transactions_by_type_2(transactions, previous_month):
    '''
    Will test the get_transactions_by_type function by requesting transactions of type debit.
    '''
    start_date = previous_month[0]
    end_date = previous_month[1]

    debits_list = dataaccess.get_transactions_by_type(start_date, end_date, transactions, 'debit')
    assert debits_list != None


if __name__ == '__main__':
    # Method names that represent tests must start with the string 'test'.
    # This naming convention informs the test runner which methods represent tests.
    #unittest.main()

    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataAccess)
    unittest.TextTestRunner(verbosity=2).run(suite)
