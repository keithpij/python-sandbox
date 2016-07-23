#!
# User Analytics
import re
import datetime
import gzip
import awstools
import MyDataTools


class User:
    def __init__(self, userDetailsList):
        self.UserId = userDetailsList[0].strip()
        self.Email = userDetailsList[1].strip()
        self.RecieveEmails = userDetailsList[2].strip()
        self.FullName = userDetailsList[3].strip()
        self.CreatedOn = MyDataTools.toDate(userDetailsList[4])
        self.LastSeenOn = MyDataTools.toDate(userDetailsList[5])
        self.ActiveAccountMember = userDetailsList[6].strip()
        self.ActiveAccountsCount = MyDataTools.toInt(userDetailsList[7])
        self.TotalAccountsCount = MyDataTools.toInt(userDetailsList[8])
        self.ActiveAccountsNotUpgradedCount = MyDataTools.toInt(userDetailsList[9])
        self.LatestActiveAdminAccountName = userDetailsList[10].strip()
        self.LatestActiveAdminAccountId = userDetailsList[11].strip()
        self.LatestActiveAdminNotUpgradedAccountId = userDetailsList[12].strip()
        self.UsageDaysCount = MyDataTools.toInt(userDetailsList[13])
        self.HadAccount = userDetailsList[14].strip()
        self.ActiveReposCount = MyDataTools.toInt(userDetailsList[15])
        self.ActiveClassicReposCount = MyDataTools.toInt(userDetailsList[16])
        self.ActivePlatformReposCount = MyDataTools.toInt(userDetailsList[17])


def getUserCount(exportDate, users):
    count = 0
    for id in users:
        count = count + 1
    return count


def loadUserFile(file):

    # Read through each line of the file.
    count = 0
    lineCount = 0
    users = dict()
    fhand = gzip.open(file)
    for line in fhand:
        lineCount = lineCount + 1

        # The first line contains the column headers.
        if lineCount == 1:
            continue

        line = line.strip()
        detailsList = line.split(',')
        u = User(detailsList)
        users[u.UserId] = u
        count = count + 1

    return accounts, count
