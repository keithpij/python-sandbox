'''
Experiments with sentiment analysis.
'''
import os
import pyprind
import pandas as pd
import numpy as np


def create_csv():
    pbar = pyprind.ProgBar(50000)
    labels = {'pos':1, 'neg':0}
    data_frame = pd.DataFrame()

    for s in ('test', 'train'):
        for l in ('pos', 'neg'):
            path = os.path.join('.', 'aclimdb', s, l) #'./aclimdb/%s/%s' % (s, l)
            for file in os.listdir(path):
                with open(os.path.join(path, file), 'r') as infile:
                    txt = infile.read()
                    data_frame = data_frame.append([[txt, labels[l]]], ignore_index=True)
                    pbar.update()

    data_frame.columns = ['review', 'sentiment']
    np.random.seed(0)
    data_frame = data_frame.reindex(np.random.permutation(data_frame.index))
    data_frame.to_csv(os.path.join('.', 'movie_data.csv'), index=False)


def read_csv():
    data_frame = pd.read_csv(os.path.join('.', 'movie_data.csv'))
    return data_frame.head(5)


if __name__ == '__main__':
    #create_csv()
    df = read_csv()
    print(df)