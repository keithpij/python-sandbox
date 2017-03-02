

if __name__ == '__main__':
    import os
    import sys
    import inspect
    import coverage
    import unittest

    # sys.path needs to be updated before any of the tests are loaded.
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)

    # Start code coverage tracking.
    # Code coverage tracking needs to start before the first import modules that contain unit tests.
    # If code coverage is started after unit test have been imported then all the code that runs as a
    # result of the import statements will be missed.
    cov = coverage.Coverage()
    cov.erase()
    cov.start()

    import test_data_access
    import test_print_tools

    # Setup the unit tests and suites.
    data_access_suite = unittest.TestLoader().loadTestsFromTestCase(test_data_access.TestDataAccess)
    print_suite = unittest.TestLoader().loadTestsFromTestCase(test_print_tools.TestPrintTools)
    all_test = unittest.TestSuite([data_access_suite, print_suite])

    unittest.TextTestRunner(stream=None, descriptions=True, verbosity=2, failfast=False, buffer=True,
        resultclass=None, warnings=None, tb_locals=False).run(all_test)

    # End coverage tracking and save the results.
    cov.stop()
    cov.save()

    cov.html_report()
    cov.report()
