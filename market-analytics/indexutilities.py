# Index Models
import os
import glob
import gzip
import shutil
import datatools
import googlecloudstorage2
import models


PRICING_BUCKET_NAME = 'keithpij-market-analytics'


def loaddatafromblobs():
    # Get a list of blobs in the companies bucket.
    blobsindex = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'INDEX')

    # Initialize the index dictionary.
    indexDictionary = dict()

    # Loop through each blob.
    for blob in blobsindex:
        print(blob.name)
        parseblob(indexDictionary, blob)

    return indexDictionary


def parseblob(indexDictionary, blob):
    handle = googlecloudstorage2.download_blob_as_bytes(PRICING_BUCKET_NAME, blob.name)
    handle.seek(0)
    gzip_file_handle = gzip.GzipFile(fileobj=handle, mode='r')

    # Loop through each line.
    for bline in gzip_file_handle:
        line = bline.decode()
        line = line.strip()
        indexList = line.split(',')

        i = Index(indexList)

        # Only interested in Dow Jones and NASDAQ.
        if i.Symbol != 'DJI' and i.Symbol != 'NAST':
            continue

        if i.Symbol not in indexDictionary:
            dateDictionary = dict()
            dateDictionary[i.Date] = i
            indexDictionary[i.Symbol] = dateDictionary
        else:
            dateDictionary = indexDictionary[i.Symbol]
            dateDictionary[i.Date] = i

    # Close the stream
    gzip_file_handle.close()


def PrintLastDateForIndex(marketData, symbol):

    daysDict = marketData[symbol.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    t = daysDict[lastDay]

    # Create the date display.
    date = str(t.Date)
    displaydate = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
    change = float(t.Close) - float(t.Open)

    # create the display string
    d = t.Symbol + '\t' + displaydate + '\t'
    d = d + '{:7,.2f}'.format(float(t.Open)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.High)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Low)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Close)) + '\t'
    d = d + '{:7,.2f}'.format(change) + '\t'
    d = d + '{:11,.0f}'.format(int(t.Volume))
    print(d)
