#!
# User Models
import re
import datetime
import gzip
import awstools
import DataTools
import csv


class User:

    FIELD_COUNT = 0

    USERID_HEADER = 'user_id'
    EMAIL_HEADER = 'email'
    RECEIVEEMAIL_HEADER = 'receive_emails'
    FULLNAME_HEADER = 'full_name'
    CREATEDON_HEADER = 'created_on'
    LASTSEENON_HEADER = 'last_seen_on'
    ACTIVEACCOUNTMEMBER_HEADER = 'active_account_member'
    ACTIVEACCOUNTSCOUNT_HEADER = 'active_accounts_count'
    TOTALACCOUNTSCOUNT_HEADER = 'total_accounts_count'
    AANOTUPGRADEDCOUNT_HEADER = 'active_accounts_not_upgraded_count'
    LATESTAAACOUNTNAME_HEADER = 'latest_active_admin_account_name'
    LATESTAAAID_HEADER = 'latest_active_admin_account_id'
    LATESTAANUAID_HEADER = 'latest_active_admin_not_upgraded_account_id'
    USAGEDAYSCOUNT_HEADER = 'usage_days_count'
    HADACCOUNT_HEADER = 'had_account'
    ACTIVEREPOSCOUNT_HEADER = 'active_repos_count'
    ACTIVECLASSICREPOSCOUNT_HEADER = 'active_classic_repos_count'
    ACTIVEPLATFORMREPOSCOUNT_HEADER = 'active_platform_repos_count'

    IndexDict = dict()

    def __init__(self, userDetailsList):

        d = User.IndexDict
        self.UserId = userDetailsList[d[User.USERID_HEADER]].strip()
        self.Email = userDetailsList[d[User.EMAIL_HEADER]].strip()
        self.RecieveEmails = userDetailsList[d[User.RECEIVEEMAIL_HEADER]].strip()
        self.FullName = userDetailsList[d[User.FULLNAME_HEADER]].strip()
        self.CreatedOn = DataTools.toDate(userDetailsList[d[User.CREATEDON_HEADER]])
        self.LastSeenOn = DataTools.toDate(userDetailsList[d[User.LASTSEENON_HEADER]])
        self.ActiveAccountMember = userDetailsList[d[User.ACTIVEACCOUNTMEMBER_HEADER]].strip()
        self.ActiveAccountsCount = DataTools.toInt(userDetailsList[d[User.ACTIVEACCOUNTSCOUNT_HEADER]])
        self.TotalAccountsCount = DataTools.toInt(userDetailsList[d[User.TOTALACCOUNTSCOUNT_HEADER]])
        self.ActiveAccountsNotUpgradedCount = DataTools.toInt(userDetailsList[d[User.AANOTUPGRADEDCOUNT_HEADER]])
        self.LatestActiveAdminAccountName = userDetailsList[d[User.LATESTAAACOUNTNAME_HEADER]].strip()
        self.LatestActiveAdminAccountId = userDetailsList[d[User.LATESTAAAID_HEADER]].strip()
        self.LatestActiveAdminNotUpgradedAccountId = userDetailsList[d[User.LATESTAANUAID_HEADER]].strip()
        self.UsageDaysCount = DataTools.toInt(userDetailsList[d[User.USAGEDAYSCOUNT_HEADER]])
        self.HadAccount = userDetailsList[d[User.HADACCOUNT_HEADER]].strip()
        self.ActiveReposCount = DataTools.toInt(userDetailsList[d[User.ACTIVEREPOSCOUNT_HEADER]])
        self.ActiveClassicReposCount = DataTools.toInt(userDetailsList[d[User.ACTIVECLASSICREPOSCOUNT_HEADER]])
        self.ActivePlatformReposCount = DataTools.toInt(userDetailsList[d[User.ACTIVEPLATFORMREPOSCOUNT_HEADER]])


def setIndecies(headerList):
    # The field count from the header line will help to correct scenarios
    # where the user put a comma in the Account Name field.
    global FIELD_COUNT
    FIELD_COUNT = len(headerList)

    # By not hard coding these value we are insulated from changes to the
    # underlying export file.
    for i in range(0, FIELD_COUNT):
        User.IndexDict[headerList[i]] = i


def getUserCount(exportDate, users):
    count = 0
    for id in users:
        if users[id].ActiveAccountsCount > 0:
            count = count + 1
    return count


def loadUserFile(file):

    # Read through each line of the file.
    count = 0
    lineCount = 0
    users = dict()
    fhand = gzip.open(file)
    reader = csv.reader(fhand)
    for line in reader:
        lineCount = lineCount + 1

        # The first line contains the column headers.
        if lineCount == 1:
            setIndecies(line)
        else:
            u = User(line)
            users[u.UserId] = u
            count = count + 1

    return users, count
