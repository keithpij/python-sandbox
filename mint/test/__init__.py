'''
Initialization file for the test directory.
'''
import os
import sys
import inspect

print('I am in the mint.test.__init__ file people.')

# sys.path needs to be updated before any of the tests are loaded.
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
