'''
Market pricing classes and functions.
'''
import os
import glob
import gzip
import shutil
import DataTools
import googlecloudstorage2

PRICING_BUCKET_NAME = 'keithpij-market-analytics'


class Price:

    def __init__(self, priceList):

        symbolIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = priceList[symbolIndex]
        date = priceList[dateIndex]
        openPrice = priceList[openIndex]
        high = priceList[highIndex]
        low = priceList[lowIndex]
        close = priceList[closeIndex]
        volume = priceList[volumeIndex]

        self.Symbol = symbol
        self.Date = DataTools.toDate(date)
        self.Open = float(openPrice)
        self.High = float(high)
        self.Low = float(low)
        self.Close = float(close)
        self.Change = self.Close - self.Open
        self.Volume = int(volume)


def ziptextfiles():
    DATA_DIR = '/Users/keithpij/Documents/eod-data'
    pricingDir = os.path.join(DATA_DIR, 'pricing')

    pricingzippeddir = os.path.join(DATA_DIR, 'pricing-zipped')
    if not os.path.isdir(pricingzippeddir):
        os.mkdir(pricingzippeddir)
        print('Creating ' + pricingzippeddir)

    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt')
    indexDirSearch = os.path.join(pricingDir, 'INDEX*.txt')

    # Notice how lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    indexFiles = glob.glob(indexDirSearch)
    allfiles = nasdaqFiles + nyseFiles + indexFiles

    for file in allfiles:
        gzfile = os.path.basename(file) + '.gz'
        gzfile = os.path.join(pricingzippeddir, gzfile)
        if not os.path.isfile(gzfile):
            fhandin = open(file, 'rb')
            fhandout = gzip.open(gzfile, 'wb')
            shutil.copyfileobj(fhandin, fhandout)
            print(gzfile)
            fhandout.close()
            fhandin.close()


def uploadfiles():
    DATA_DIR = '/Users/keithpij/Documents/eod-data'

    pricingzippeddir = os.path.join(DATA_DIR, 'pricing-zipped')
    if not os.path.isdir(pricingzippeddir):
        return

    nasdaqDirSearch = os.path.join(pricingzippeddir, 'NASDAQ*.txt.gz')
    nyseDirSearch = os.path.join(pricingzippeddir, 'NYSE*.txt.gz')
    indexDirSearch = os.path.join(pricingzippeddir, 'INDEX*.txt.gz')

    # Notice how lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    indexFiles = glob.glob(indexDirSearch)
    allfiles = nasdaqFiles + nyseFiles + indexFiles

    for file in allfiles:
        blobname = os.path.basename(file)
        googlecloudstorage2.upload_blob('keithpij-market-analytics', file, blobname)
        print(blobname)


def loaddatafromblobs():
    # Get a list of blobs in the companies bucket.
    blobsnasdaq = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NASDAQ')
    blobsnyse = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NYSE')

    # Initialize the pricing dictionary.
    pricingDictionary = dict()

    # NASDAQ
    for blob in blobsnasdaq:
        print(blob.name)
        parseblob(pricingDictionary, blob)

    # NYSE
    for blob in blobsnyse:
        print(blob.name)
        parseblob(pricingDictionary, blob)

    return pricingDictionary


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
    pricingDir = os.path.join(DATA_DIR, 'eod-data', 'pricing-zipped')
    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt.gz')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt.gz')

    # Notice how two lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    allFiles = nasdaqFiles + nyseFiles

    pricingDictionary = parsefiles(allFiles)

    return pricingDictionary


def parsefiles(files):
    pricingDictionary = dict()
    for file in files:

        # Read through each line of the file.
        fhand = gzip.open(file, 'r')
        for bline in fhand:

            # The line is still in binary.
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

        # Close the file
        fhand.close()

    return pricingDictionary


if __name__ == '__main__':
    #ziptextfiles()
    #uploadfiles()

    # Get a list of blobs in the companies bucket.
    blobsnasdaq = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NASDAQ')
    blobsnyse = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NYSE')
    blobs = blobsnasdaq + blobsnyse
    print(blobs)

    '''
    handle = googlecloudstorage2.download_blob_as_bytes('keithpij-market-analytics', 'NYSE_20160909.txt.gz')
    handle.seek(0)
    gzip_file_handle = gzip.GzipFile(fileobj=handle, mode='r')
    for bline in gzip_file_handle:
        line = bline.decode()
        line = line.strip()
        priceList = line.split(',')

        p = Price(priceList)
        print(p.Symbol)

    gzip_file_handle.close()
    '''
