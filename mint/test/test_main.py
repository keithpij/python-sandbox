import unittest
import test_data_access
import test_print_tools


if __name__ == '__main__':

    data_access_suite = unittest.TestLoader().loadTestsFromTestCase(test_data_access.TestDataAccess)
    print_suite = unittest.TestLoader().loadTestsFromTestCase(test_print_tools.TestPrintTools)
    all_test = unittest.TestSuite([data_access_suite, print_suite])
    unittest.TextTestRunner(verbosity=2).run(all_test)
