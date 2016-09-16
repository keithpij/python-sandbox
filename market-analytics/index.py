# Index Models
import os
import glob
import googlecloudstorage2


class Index:

    def __init__(self, tickerList):

        nameIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = tickerList[nameIndex]
        date = tickerList[dateIndex]
        openPrice = tickerList[openIndex]
        high = tickerList[highIndex]
        low = tickerList[lowIndex]
        close = tickerList[closeIndex]
        volume = tickerList[volumeIndex]

        self.Symbol = symbol
        self.Date = date
        self.Open = float(openPrice)
        self.High = float(high)
        self.Low = float(low)
        self.Close = float(close)
        self.Change = self.Close - self.Open
        self.Volume = int(volume)


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


def parseblob(pricingDictionary, blob):
    handle = googlecloudstorage2.download_blob_as_bytes(PRICING_BUCKET_NAME, blob.name)
    handle.seek(0)
    gzip_file_handle = gzip.GzipFile(fileobj=handle, mode='r')

    # Loop through each line.
    for bline in gzip_file_handle:
        line = bline.decode()
        line = line.strip()
        priceList = line.split(',')

        p = Price(priceList)

        if p.Symbol not in pricingDictionary:
            dateDictionary = dict()
            dateDictionary[p.Date] = p
            pricingDictionary[p.Symbol] = dateDictionary
        else:
            dateDictionary = pricingDictionary[p.Symbol]
            dateDictionary[p.Date] = p

    # Close the stream
    gzip_file_handle.close()


def loaddatafromfiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data', 'pricing')
    indexDirSearch = os.path.join(pricingDir, 'INDEX*.txt')

    # Notice how two lists can be added together.
    indexFiles = glob.glob(indexDirSearch)

    indexDictionary = parsefiles(indexFiles)

    return indexDictionary


def parsefiles(files):
    indexDictionary = dict()
    for file in files:
        #print(file)

        # Read through each line of the file.
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            indexList = line.split(',')

            # Create an instance of the Index class.
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

        # Close the file
        fhand.close()

    return indexDictionary


def PrintLastDateForIndex(marketData, tickerName):

    daysDict = marketData[tickerName.upper()]
    daysListSorted = sorted(daysDict)
    lastDay = daysListSorted[-1]
    t = daysDict[lastDay]

    # Create the date display.
    date = t.Date
    date = date[4:6] + '/' + date[6:8] + '/' + date[0:4]
    change = float(t.Close) - float(t.Open)

    # create the display string
    d = t.Symbol + '\t' + date + '\t'
    d = d + '{:7,.2f}'.format(float(t.Open)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.High)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Low)) + '\t'
    d = d + '{:7,.2f}'.format(float(t.Close)) + '\t'
    d = d + '{:7,.2f}'.format(change) + '\t'
    d = d + '{:11,.0f}'.format(int(t.Volume))
    print(d)
