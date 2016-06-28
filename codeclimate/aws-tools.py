# AWS Tools
import re
import datetime
from boto.s3.connection import S3Connection


def filesToList():
    currentWorkingDir = os.getcwd()
    pricingDir = os.path.join(currentWorkingDir, 'eod-data')
    nasdaqDirSearch = os.path.join(pricingDir, 'NASDAQ*.txt')
    nyseDirSearch = os.path.join(pricingDir, 'NYSE*.txt')

    # Notice how two lists can be added together.
    nasdaqFiles = glob.glob(nasdaqDirSearch)
    nyseFiles = glob.glob(nyseDirSearch)
    allFiles = nasdaqFiles + nyseFiles

    marketList = parseFiles(allFiles)

    return marketList


AWS_ACCESS_KEY_ID = 'AKIAJZJ5YNSTAGIUS7KQ'
AWS_SECRET_ACCESS_KEY = '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'
AWS_S3_BUCKET = 'codeclimate-exports'
DATA_DIR = '/Users/keithpijanowski/Documents/cc-data'

t = str(datetime.date.today())
y = t[0:4]
m = t[5:7]
d = t[8:10]
todayRegex = '^account_export-' + y + '-'
todayRegex = todayRegex + m + '-'
todayRegex = todayRegex + d

connection = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
bucket = connection.get_bucket(AWS_S3_BUCKET)

# Loop through all the export files.
count = 0
print('Looking for objects that match: ' + todayRegex)
for fileKey in bucket.list():
    if re.search(todayRegex, fileKey.name):
        todaysKey = fileKey
        count = count + 1
        print(fileKey.name)

# print(str(count) + ' objects in ' + AWS_S3_BUCKET)

# todaysKey.get_contents_to_filename('/home/larry/documents/perl_poetry.pdf')
