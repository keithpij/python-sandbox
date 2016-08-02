# Analytics Module
import re
import datetime
import awstools
import DataTools
import UserModels
import AccountModels


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
            c = UserModels.getUserCount(exportDate, users)
            print('Total number of users: ' + str(c))

        # Search by name.  The search text must be a regular expression.
        if command[0:2] == '-n':
            regx = command[3:]
            matches = searchByName(regx, accounts)
            printBrief(matches)

        if command[0:5] == '-load':
            s = command[6:]
            exportDate = DataTools.toDate(s)
            accounts, count = getAllData(exportDate)

        if command[0:6] == '-trial':
            matches = getTrialAccounts(exportDate, accounts)
            printBrief(matches)

        if command[0:6] == '-churn':
            d = command[7:]
            daysInPast = DataTools.toInt(d)
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

    if accountFile is not None:
        print('Loading account data ...')
        # print('Loading ' + userFile)
        accounts, aCount = AccountModels.loadAccountFile(accountFile)
        print('Total number of account records:  ' + str(aCount))
    else:
        accounts = None
        print('No account data found.')

    if userFile is not None:
        print('Loading user data ...')
        # print('Loading ' + userFile)
        users, uCount = UserModels.loadUserFile(userFile)
        print('Total number of user records:  ' + str(uCount))
    else:
        users = None
        print('No account data found.')

    return accounts, users


# Start of main processing.
exportDate = datetime.date.today()  # DataTools.toDate('2016-07-01')  DataTools.toDate('2016-06-17')
accounts, users = getAllData(exportDate)

if accounts is None:
    ed = raw_input('Specify a new export date (YYYY-MM-DD):  ')
    exportDate = DataTools.toDate(ed)
    accounts, users = getAllData(exportDate)

if accounts is not None:
    getUserRequests(exportDate, accounts, users)
