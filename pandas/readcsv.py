'''
This module will read a csv file into a Pandas dataframe.
'''
import pandas as pd


df = pd.read_csv('gsod_1929.csv')
print(df.head(10))
