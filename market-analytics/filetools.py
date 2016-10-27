'''
Market pricing classes and functions.
'''
import os
import glob
import gzip
import csv
import shutil
import datatools
import googlecloudstorage2
import models
import company


PRICING_BUCKET_NAME = 'keithpij-market-analytics'


def zipTextFiles():
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


def uploadPricingFiles():
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

    # Get the blobs in the pricing bucket and create a dictionary to
    # be used to test for the existance of the blob.
    blobs = googlecloudstorage2.get_blobs(PRICING_BUCKET_NAME)
    bd = dict()
    for blob in blobs:
        bd[blob.name] = None

    for file in allfiles:
        blobname = os.path.basename(file)

        # Determine if the file needs to be uploaded.
        if not blobname in bd:
            googlecloudstorage2.upload_blob('keithpij-market-analytics', file, blobname)
            print('Uploading: ' + blobname)


def loadPricingFromBlobs():
    # Get a list of blobs in the companies bucket.
    blobsnasdaq = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NASDAQ')
    blobsnyse = googlecloudstorage2.get_blobs_with_prefix(PRICING_BUCKET_NAME, 'NYSE')

    # Initialize the pricing dictionary.
    pricingDictionary = dict()

    # NASDAQ
    for blob in blobsnasdaq:
        print(blob.name)
        parsePricingBlob(pricingDictionary, blob)

    # NYSE
    for blob in blobsnyse:
        print(blob.name)
        parseblob(pricingDictionary, blob)

    return pricingDictionary


def parsePricingBlob(pricingDictionary, blob):
    handle = googlecloudstorage2.download_blob_as_bytes(PRICING_BUCKET_NAME, blob.name)
    handle.seek(0)
    gzip_file_handle = gzip.GzipFile(fileobj=handle, mode='r')

    # Loop through each line.
    for bline in gzip_file_handle:
        line = bline.decode()
        line = line.strip()
        priceList = line.split(',')

        p = models.Price(priceList)

        if p.Symbol not in pricingDictionary:
            dateDictionary = dict()
            dateDictionary[p.Date] = p
            pricingDictionary[p.Symbol] = dateDictionary
        else:
            dateDictionary = pricingDictionary[p.Symbol]
            dateDictionary[p.Date] = p

    # Close the stream
    gzip_file_handle.close()


def loadCompaniesFromBlobs():
    # Get a list of blobs in the companies bucket.
    blobs = googlecloudstorage2.get_blobs(COMPANIES_BUCKET_NAME)

    companyCount = 0
    companyDictionary = dict()
    for blob in blobs:
        a = blob.name.split('.')
        b = a[0]  # get rid of the file extension
        c = b.split('-')
        date = datatools.toDate(c[-3] + c[-2] + c[-1])
        print(blob.name)
        print(date)

        # Read through each line of the file.
        lineCount = 0
        filedata = googlecloudstorage2.download_blob_as_string(COMPANIES_BUCKET_NAME, blob.name)
        f = io.StringIO(filedata)

        reader = csv.reader(f)
        for line in reader:
            lineCount = lineCount + 1

            # The first line contains the column headers.
            if lineCount > 1:
                c = Company(date, line)
                companyDictionary[c.Symbol] = c
                companyCount = companyCount + 1

    return companyDictionary, companyCount


def loadPricingFiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data', 'pricing-zipped')
    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt.gz')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt.gz')

    # Notice how two lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    allFiles = nasdaqFiles + nyseFiles

    pricingDictionary = parsePricingFiles(allFiles)

    return pricingDictionary


def parsePricingFiles(files):
    pricingDictionary = dict()
    for file in files:
        print(file)

        # Read through each line of the file.
        fhand = gzip.open(file, 'r')
        for bline in fhand:

            # The line is still in binary.
            line = bline.decode()

            line = line.strip()
            priceList = line.split(',')

            p = models.Price(priceList)

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


def loadIndexFiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data', 'pricing')
    indexDirSearch = os.path.join(pricingDir, 'INDEX*.txt')

    # Notice how two lists can be added together.
    indexFiles = glob.glob(indexDirSearch)

    indexDictionary = parseIndexFiles(indexFiles)

    return indexDictionary


def parseIndexFiles(files):
    indexDictionary = dict()
    for file in files:
        #print(file)

        # Read through each line of the file.
        fhand = open(file)
        for line in fhand:

            line = line.strip()
            indexList = line.split(',')

            # Create an instance of the Index class.
            i = models.Price(indexList)

            # Only interested in Dow Jones and NASDAQ.
            #if i.Symbol != 'DJI' and i.Symbol != 'NAST':
            #    continue

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


def loadCompanyFiles():
    # currentWorkingDir = os.getcwd()
    DATA_DIR = os.path.join('/Users', 'keithpij', 'Documents')
    #DATA_DIR = '/Users/keithpij/Documents'
    pricingDir = os.path.join(DATA_DIR, 'eod-data')
    fileSearch = os.path.join(pricingDir, '*companylist*.csv')

    allFiles = glob.glob(fileSearch)

    companyDictionary = parseCompanyFiles(allFiles)

    return companyDictionary


def parseCompanyFiles(files):
    companyCount = 0
    companyDictionary = dict()
    for file in files:
        a = file.split('.')
        b = a[0]  # get rid of the file extension
        c = b.split('-')
        date = datatools.toDate(c[-3] + c[-2] + c[-1])

        # Read through each line of the file.
        lineCount = 0
        fhand = open(file)
        reader = csv.reader(fhand)
        for line in reader:
            lineCount = lineCount + 1

            # The first line contains the column headers.
            if lineCount > 1:
                c = company.Company(date, line)
                companyDictionary[c.Symbol] = c
                companyCount = companyCount + 1

    return companyDictionary


if __name__ == '__main__':
    #ziptextfiles()
    uploadfiles()
