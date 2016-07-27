# Account class
import re
import datetime
import gzip
import awstools
import MyDataTools
import useranalytics


class Account:

    FIELD_COUNT = 0
    ACCOUNTID_INDEX = -1
    ACCOUNTNAME_INDEX = -1
    FIRSTOWNERNAME_INDEX = -1
    FIRSTOWNEREMAIL_INDEX = -1
    ORGANIZATIONSIZE_INDEX = -1
    PHONENUMBER_INDEX = -1
    PROMOCODE_INDEX = -1
    USERSCOUNT_INDEX = -1
    SEATSCOUNT_INDEX = -1
    AUTHORSCOUNT_INDEX = -1
    USERQUOTA_INDEX = -1
    REPOSCOUNT_INDEX = -1
    PLATFORMREPOSCOUNT_INDEX = -1
    ACTIVEREPOSCOUNT_INDEX = -1
    ACTIVEPLATFORMREPOSCOUNT_INDEX = -1
    CREATEDON_INDEX = -1
    TRIALENDSON_INDEX = -1
    PLANNAME_INDEX = -1
    PLANPRICE_INDEX = -1
    PLANCODE_INDEX = -1
    BILLINGINTERVAL_INDEX = -1
    SUBSCRIPTIONPERIODSTART_INDEX = -1
    SUBSCRIPTIONPERIODEND_INDEX = -1
    UTMCAMPAIGN_INDEX = -1

    def __init__(self, accountDetailsList):

        # accountDetailsList is a List that should have the same number of
        # entries as the header line.
        # If it has more, than there was a comma in the account name field.
        if len(accountDetailsList) != FIELD_COUNT:
            # print(accountDetailsList[0] + ' ' + str(len(accountDetailsList)))
            accountDetailsList = fixList(accountDetailsList)

        # print(accountDetailsList)
        self.AccountId = accountDetailsList[ACCOUNTID_INDEX]
        self.AccountName = accountDetailsList[ACCOUNTNAME_INDEX].strip()
        self.FirstOwnerName = accountDetailsList[FIRSTOWNERNAME_INDEX].strip()
        self.FirstOwnerEmail = accountDetailsList[FIRSTOWNEREMAIL_INDEX].strip()
        self.OrganizationSize = accountDetailsList[ORGANIZATIONSIZE_INDEX].strip()
        self.PhoneNumber = accountDetailsList[PHONENUMBER_INDEX].strip()
        self.PromoCode = accountDetailsList[PROMOCODE_INDEX].strip()
        self.UsersCount = MyDataTools.toInt(accountDetailsList[USERSCOUNT_INDEX])  # this could change over time
        self.SeatCount = MyDataTools.toInt(accountDetailsList[SEATSCOUNT_INDEX])
        self.AuthorsCount = MyDataTools.toInt(accountDetailsList[AUTHORSCOUNT_INDEX])
        self.UserQuota = MyDataTools.toInt(accountDetailsList[USERQUOTA_INDEX])
        self.ReposCount = MyDataTools.toInt(accountDetailsList[REPOSCOUNT_INDEX])  # this could change over time
        self.PlatformReposCount = MyDataTools.toInt(accountDetailsList[PLATFORMREPOSCOUNT_INDEX])
        self.ActiveReposCount = MyDataTools.toInt(accountDetailsList[ACTIVEREPOSCOUNT_INDEX])
        self.ActivePlatformReposCount = MyDataTools.toInt(accountDetailsList[ACTIVEPLATFORMREPOSCOUNT_INDEX])
        self.CreatedOn = accountDetailsList[CREATEDON_INDEX]
        self.TrialEndsOn = MyDataTools.toDate(accountDetailsList[TRIALENDSON_INDEX])
        self.PlanName = accountDetailsList[PLANNAME_INDEX]
        self.PlanPrice = accountDetailsList[PLANPRICE_INDEX]
        self.PlanCode = accountDetailsList[PLANCODE_INDEX]
        self.BillingInterval = accountDetailsList[BILLINGINTERVAL_INDEX]
        self.SubscriptionPeriodStart = MyDataTools.toDate(accountDetailsList[SUBSCRIPTIONPERIODSTART_INDEX])
        self.SubscriptionPeriodEnd = MyDataTools.toDate(accountDetailsList[SUBSCRIPTIONPERIODEND_INDEX])
        self.UtmCampaign = accountDetailsList[UTMCAMPAIGN_INDEX]
        # Need fields for:
        # credit card churn date
        # customer cancellation date
        # cancellation reason
        # SCM
        # Platform Repo count
        # Classic Repo count


def fixList(l):
    length = len(l)
    delta = length - FIELD_COUNT

    # Initialize the new list.
    newList = list()
    for i in range(0, FIELD_COUNT-1):
        newList.append('')

    # Everything up to the Account name index is fine.
    # This should just be the Account Name field.
    for i in range(0, ACCOUNTNAME_INDEX):
        newList[i] = l[i]

    # Fix up the account name field.
    for i in range(ACCOUNTNAME_INDEX, ACCOUNTNAME_INDEX + delta + 1):
        if i == ACCOUNTNAME_INDEX:
            newList[ACCOUNTNAME_INDEX] = l[i]
        else:
            newList[ACCOUNTNAME_INDEX] = newList[ACCOUNTNAME_INDEX] + ', ' + l[i]

    # Get everything after the account name field.
    for i in range(ACCOUNTNAME_INDEX + delta + 1, length-1):
        newList[i-delta] = l[i]

    print(delta)
    print(l)
    print(newList)

    return newList


def setIndecies(headerList):
    # The field count from the header line will help to correct scenarios
    # where the user put a comma in the Account Name field.
    global FIELD_COUNT
    FIELD_COUNT = len(headerList)

    # By not hard coding these value we are insulated from changes to the
    # underlying export file.
    for i in range(0, FIELD_COUNT-1):
        h = headerList[i]
        if h == 'account_id':
            Account.ACCOUNTID_INDEX = i
        elif h == 'account_name':
            Account.ACCOUNTNAME_INDEX = i
        elif h == 'first_owner_name':
            Account.FIRSTOWNERNAME_INDEX = i
        elif h == 'first_owner_email':
            Account.FIRSTOWNEREMAIL_INDEX = i
        elif h == 'organization_size':
            Account.ORGANIZATIONSIZE_INDEX = i
        elif h == 'phone_number':
            Account.PHONENUMBER_INDEX = i
        elif h == 'promo_code':
            Account.PROMOCODE_INDEX = i
        elif h == 'users_count':
            Account.USERSCOUNT_INDEX = i
        elif h == 'seats_count':
            Account.SEATSCOUNT_INDEX = i
        elif h == 'authors_count':
            Account.AUTHORSCOUNT_INDEX = i
        elif h == 'user_quota':
            Account.USERQUOTA_INDEX = i
        elif h == 'repos_count':
            Account.REPOSCOUNT_INDEX = i
        elif h == 'platform_repos_count':
            Account.PLATFORMREPOSCOUNT_INDEX = i
        elif h == 'active_repos_count':
            Account.ACTIVEREPOSCOUNT_INDEX = i
        elif h == 'active_platform_repos_count':
            Account.ACTIVEPLATFORMREPOSCOUNT_INDEX = i
        elif h == 'created_on':
            Account.CREATEDON_INDEX = i
        elif h == 'trial_ends_on':
            Account.TRIALENDSON_INDEX = i
        elif h == 'plan_name':
            Account.PLANNAME_INDEX = i
        elif h == 'plan_price':
            Account.PLANPRICE_INDEX = i
        elif h == 'plan_code':
            Account.PLANCODE_INDEX = i
        elif h == 'billing_interval':
            Account.BILLINGINTERVAL_INDEX = i
        elif h == 'subscription_period_start':
            Account.SUBSCRIPTIONPERIODSTART_INDEX = i
        elif h == 'subscription_period_end':
            Account.SUBSCRIPTIONPERIODEND_INDEX = i
        elif h == 'utm_campaign':
            Account.UTMCAMPAIGN_INDEX = i


def loadAccountFile(file):

    # Setup needed variables
    accountCount = 0
    lineCount = 0
    accounts = dict()
    fhand = gzip.open(file)

    # Read through each line of the file.
    for line in fhand:
        lineCount = lineCount + 1

        line = line.strip()
        detailsList = line.split(',')
        # The first line contains the column headers.
        if lineCount == 1:
            setIndecies(detailsList)
        else:
            a = Account(detailsList)
            accounts[a.AccountId] = a
            accountCount = accountCount + 1

    return accounts, accountCount


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
exportDate = datetime.date.today()  # MyDataTools.toDate('2016-07-01')
accounts, users = getAllData(exportDate)

if accounts is None:
    ed = raw_input('Specify a new export date (YYYY-MM-DD):  ')
    exportDate = MyDataTools.toDate(ed)
    accounts, users = getAllData(exportDate)

if accounts is not None:
    getUserRequests(exportDate, accounts, users)
