# Account class
import re
import datetime
import gzip
import awstools
import MyDataTools
import useranalytics


class Account:

    def __init__(self, accountDetailsList):

        # accountDetailsList is a List that should have 17 entries.
        # If it has more than there was a comma in the account name field.
        if len(accountDetailsList) >= 19:
            # print(accountDetailsList[0] + ' ' + str(len(accountDetailsList)))
            accountDetailsList = fixList(accountDetailsList)

        # print(accountDetailsList)
        self.AccountId = accountDetailsList[0]
        self.AccountName = accountDetailsList[1].strip()
        self.FirstOwnerName = accountDetailsList[2].strip()
        self.FirstOwnerEmail = accountDetailsList[3].strip()
        self.OrganizationSize = accountDetailsList[4].strip()
        self.PhoneNumber = accountDetailsList[5].strip()
        self.PromoCode = accountDetailsList[6].strip()
        self.UsersCount = MyDataTools.toInt(accountDetailsList[7])  # this could change over time
        self.ReposCount = MyDataTools.toInt(accountDetailsList[8])  # this could change over time
        self.CreatedOn = accountDetailsList[9]
        self.TrialEndsOn = MyDataTools.toDate(accountDetailsList[10])
        self.PlanName = accountDetailsList[11]
        self.PlanPrice = accountDetailsList[12]
        self.PlanCode = accountDetailsList[13]
        self.BillingInterval = accountDetailsList[14]
        self.SubscriptionPeriodStart = MyDataTools.toDate(accountDetailsList[15])
        self.SubscriptionPeriodEnd = MyDataTools.toDate(accountDetailsList[16])
        self.UtmCampaign = accountDetailsList[17]
        # Need fields for:
        # credit card churn date
        # customer cancellation date
        # cancellation reason
        # SCM
        # Platform Repo count
        # Classic Repo count


def fixList(l):
    length = len(l)

    # Create 18 entries in the new list.
    newList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    newList[17] = l[length-1]   # This is the last index
    newList[16] = l[length-2]
    newList[15] = l[length-3]
    newList[14] = l[length-4]
    newList[13] = l[length-5]
    newList[12] = l[length-6]
    newList[11] = l[length-7]
    newList[10] = l[length-8]
    newList[9] = l[length-9]
    newList[8] = l[length-10]
    newList[7] = l[length-11]
    newList[6] = l[length-12]
    newList[5] = l[length-13]
    newList[4] = l[length-14]
    newList[3] = l[length-15]
    newList[2] = l[length-16]

    # Account name field is the problematic field.
    # The for loop will not iterate on the final value.
    # So this needs to be length-16 and not length-17.
    for i in range(1, length-16):
        if i == 1:
            newList[1] = l[1]
        else:
            newList[1] = newList[1] + ', ' + l[length-17]

    # Account id field.  This should be a guid.
    newList[0] = l[0]

    return newList


def loadAccountFile(file):

    # Read through each line of the file.
    count = 0
    lineCount = 0
    accounts = dict()
    fhand = gzip.open(file)
    for line in fhand:
        lineCount = lineCount + 1

        # The first line contains the column headers.
        if lineCount == 1:
            continue

        line = line.strip()
        detailsList = line.split(',')
        a = Account(detailsList)
        accounts[a.AccountId] = a
        count = count + 1

    return accounts, count


def pad(field, width):
    f = str(field)
    l = len(f)

    if l >= width:
        return f[0:width-1] + ' '

    d = width - l
    for i in range(d):
        f = f + ' '
    return f


def printBrief(matches):

    header = pad('Id', 30) + pad('Acc. Name', 15) + pad('User Count', 12)
    header = header + pad('Repo Count', 12) + pad('Created On', 12)
    header = header + pad('Trial Ends', 12) + pad('Sub Starts', 12)
    header = header + pad('Sub Ends', 12)
    print(header)

    count = 0
    for id in matches:
        count = count + 1
        a = matches[id]
        d = pad(a.AccountId, 30) + pad(a.AccountName, 15) + pad(a.UsersCount, 12)
        d = d + pad(a.ReposCount, 12) + pad(a.CreatedOn, 12)
        d = d + pad(a.TrialEndsOn, 12) + pad(a.SubscriptionPeriodStart, 12)
        d = d + pad(a.SubscriptionPeriodEnd, 12)
        print(d)
    print('')
    print(str(count) + ' accounts')


def printFull(matches):
    for id in matches:
        account = matches[id]
        print('Account Id:\t' + account.AccountId)
        print('Owner Name:\t' + account.FirstOwnerName)
        print('Owner Email:\t' + account.FirstOwnerEmail)
        print('Organization Size:\t' + account.OrganizationSize)
        print('Phone:\t' + account.PhoneNumber)
        print('Promo Code:\t' + account.PromoCode)
        print('User Count:\t' + str(account.UsersCount))
        print('Repo Count:\t' + str(account.ReposCount))
        print('Created On:\t' + account.CreatedOn)
        print('Trial ends:\t' + str(account.TrialEndsOn))
        print('Plan Name:\t' + account.PlanName)
        print('Plan Price:\t' + account.PlanPrice)
        print('Plan Code:\t' + account.PlanCode)
        print('Billing Interval:\t' + account.BillingInterval)
        print('Subscription Start:\t' + str(account.SubscriptionPeriodStart))
        print('Subscription End:\t' + str(account.SubscriptionPeriodEnd))
        print('Utm Campaign:\t' + account.UtmCampaign)


def searchByName(rexp, accounts):
    matches = dict()
    for id in accounts:
        a = accounts[id]
        if re.search(rexp, a.AccountName.lower()):
            matches[id] = a
    return matches


def searchById(rexp, accounts):
    matches = dict()
    for id in accounts:
        if re.search(rexp, id.lower()):
            matches[id] = accounts[id]
    return matches


def getCustomerCount(exportDate, accounts):
    count = 0
    for id in accounts:
        a = accounts[id]
        if a.SubscriptionPeriodStart is not None:
            if a.SubscriptionPeriodEnd >= exportDate:
                count = count + 1
    return count


def getChurnedAccounts(exportDate, daysInPast, accounts):
    daysInPastDelta = datetime.timedelta(days=30)
    fromDate = exportDate - daysInPastDelta
    matches = dict()
    for id in accounts:
        a = accounts[id]
        if a.SubscriptionPeriodEnd is None:
            continue
        if a.SubscriptionPeriodEnd < fromDate:
            continue
        # Use yesterday to determine churn.  This elimanates account whose
        # subscription expires today but have not had their credit card proessed.
        yesterday = exportDate - datetime.timedelta(days=1)
        if a.SubscriptionPeriodEnd <= yesterday:
            matches[id] = accounts[id]
    return matches


def getTrialAccounts(exportDate, accounts):
    matches = dict()
    for id in accounts:
        a = accounts[id]
        if a.TrialEndsOn >= exportDate:
            matches[id] = accounts[id]
    return matches


def getUserRequests(exportDate, accounts, users):
    # User request loop
    while True:
        command = raw_input(str(exportDate) + ' --> ')

        # Quit the input loop.
        if command == '-q':
            break

        if command[0:3] == '-cc':
            c = getCustomerCount(exportDate, accounts)
            print('Total number of customers: ' + str(c))

        if command[0:3] == '-uc':
            c = useranalytics.getUserCount(exportDate, accounts)
            print('Total number of users: ' + str(c))

        # Search by name.  The search text must be a regular expression.
        if command[0:2] == '-n':
            regx = command[3:]
            matches = searchByName(regx, accounts)
            printBrief(matches)

        if command[0:5] == '-load':
            s = command[6:]
            exportDate = MyDataTools.toDate(s)
            accounts, count = getAllData(exportDate)

        if command[0:6] == '-trial':
            matches = getTrialAccounts(exportDate, accounts)
            printBrief(matches)

        if command[0:6] == '-churn':
            d = command[7:]
            daysInPast = MyDataTools.toInt(d)
            if daysInPast == 0:
                daysInPast = 30
            matches = getChurnedAccounts(exportDate, daysInPast, accounts)
            printBrief(matches)

        if command[0:3] == '-id':
            regx = command[4:]
            print(regx)
            matches = searchById(regx, accounts)
            printFull(matches)


def getAllData(exportDate):
    # Load the file containing all the accounts.
    accountFile = awstools.getAccountFile(exportDate)
    userFile = awstools.getUserFile(exportDate)

    if accountFile is None or userFile is None:
        if awstools.checkInternet():
            print('Downloading account and user exports ...')
            awstools.getExports(exportDate)
            accountFile = awstools.getAccountFile(exportDate)
            userFile = awstools.getUserFile(exportDate)
        else:
            print('Not connected to the internet.  Cannot download account and user exports.')

    if accountFile is not None and userFile is not None:
        print('Loading ...')
        # print('Loading ' + userFile)
        accounts, aCount = loadAccountFile(accountFile)
        # users, uCount = useranalytics.loadUserFile(userFile)
        print('Number of accounts:  ' + str(aCount))
        # print('Number of Users:  ' + str(uCount))
        return accounts, None
    else:
        print('No data found.')
        return None, None


# Start of main processing.
exportDate = datetime.date.today() # MyDataTools.toDate('2016-07-01')
accounts, users = getAllData(exportDate)
if accounts is not None:
    getUserRequests(exportDate, accounts, users)
