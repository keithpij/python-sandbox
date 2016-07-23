# AWS Tools
import os
import re
import datetime
import glob
import urllib2
from boto.s3.connection import S3Connection
import boto.ec2
from pprint import pprint

AWS_S3_BUCKET = 'codeclimate-exports'
DATA_DIR = '/Users/keithpijanowski/Documents/cc-data'
CC_CREDENTIALS_FILE = os.path.join(DATA_DIR, 'aws-credentials.csv')
KP_CREDENTIALS_FILE = os.path.join(DATA_DIR, 'keithpij-credentials.csv')


def checkInternet():
    try:
        response = urllib2.urlopen('http://www.google.com', timeout=1)
        return True
    except urllib2.URLError as err:
        return False


def getAWSCredentials(file):
    fhand = open(file)
    header = fhand.readline()
    credentials = fhand.readline()
    credentials = credentials.strip()
    l = credentials.split(',')

    userName = l[0]
    accessKeyId = l[1]
    secretAccessKey = l[2]

    return accessKeyId, secretAccessKey


def getAccountFile(exportDate):
    d = str(exportDate)
    y = d[0:4]
    m = d[5:7]
    d = d[8:10]
    search = 'account_export-' + y + '-'
    search = search + m + '-'
    search = search + d + '*.csv.gz'

    print('Looking for: ' + search)
    dirSearch = os.path.join(DATA_DIR, search)
    files = glob.glob(dirSearch)
    if len(files) > 0:
        return files[0]
    else:
        return None


def getUserFile(exportDate):
    d = str(exportDate)
    y = d[0:4]
    m = d[5:7]
    d = d[8:10]
    search = 'user_export-' + y + '-'
    search = search + m + '-'
    search = search + d + '*.csv.gz'

    dirSearch = os.path.join(DATA_DIR, search)
    files = glob.glob(dirSearch)
    if len(files) > 0:
        return files[0]
    else:
        return None


def getExports(exportDate):
    d = str(exportDate)
    y = d[0:4]
    m = d[5:7]
    d = d[8:10]

    # Regular expression to get today's account file.
    accountRegex = '^account_export-' + y + '-'
    accountRegex = accountRegex + m + '-'
    accountRegex = accountRegex + d

    # Regular expression to get today's account file.
    userRegex = '^user_export-' + y + '-'
    userRegex = userRegex + m + '-'
    userRegex = userRegex + d

    # Connect to AWS S3
    (awsAccessKeyId, awsSecretAccessKey) = getAWSCredentials(CC_CREDENTIALS_FILE)
    connection = S3Connection(awsAccessKeyId, awsSecretAccessKey)
    bucket = connection.get_bucket(AWS_S3_BUCKET)

    # Loop through all the export files.
    count = 0
    accountKey = None
    userKey = None
    print('Looking for todays account and user exports.')
    for fileKey in bucket.list():
        count = count + 1
        if re.search(accountRegex, fileKey.name):
            accountKey = fileKey
        if re.search(userRegex, fileKey.name):
            userKey = fileKey


    # TODO - check local file system before getting a list of objects from S3.
    if accountKey is not None:
        f = os.path.join(DATA_DIR, accountKey.name)
        if not os.path.exists(f):
            print('Downloading ' + accountKey.name)
            accountKey.get_contents_to_filename(f)
        else:
            print(f + ' already exists')

    if userKey is not None:
        f = os.path.join(DATA_DIR, userKey.name)
        if not os.path.exists(f):
            print('Downloading ' + userKey.name)
            accountKey.get_contents_to_filename(f)
        else:
            print(f + ' already exists')


def getEC2Instances(region):
    # regions = ['us-east-1', 'us-west-1', 'us-west-2', 'eu-west-1', 'eu-central-1', 'sa-east-1',
    #         'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1', 'ap-northeast-2', 'ap-south-1']

    (awsAccessKeyId, awsSecretAccessKey) = getAWSCredentials(KP_CREDENTIALS_FILE)

    # for region in regions:

    ec2_conn = boto.ec2.connect_to_region(region,
            aws_access_key_id=awsAccessKeyId,
            aws_secret_access_key=awsSecretAccessKey)

    reservations = ec2_conn.get_all_reservations()
    for reservation in reservations:
        # print(region+':', reservation.instances)

        for i in reservation.instances:
            pprint(i.__dict__)
            i.stop()

    # for vol in ec2_conn.get_all_volumes():
    #    print(region+':', vol.id)


# Main execution
# getEC2Instances('us-east-1')
