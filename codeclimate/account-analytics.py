# Account class
import re
import datetime

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
        self.UsersCount = toInt(accountDetailsList[7])
        self.ReposCount = toInt(accountDetailsList[8])
        self.CreatedOn = accountDetailsList[9]
        self.TrialEndsOn = toDate(accountDetailsList[10])
        self.PlanName = accountDetailsList[11]
        self.PlanPrice = accountDetailsList[12]
        self.PlanCode = accountDetailsList[13]
        self.BillingInterval = accountDetailsList[14]
        self.SubscriptionPeriodStart = toDate(accountDetailsList[15])
        self.SubscriptionPeriodEnd = toDate(accountDetailsList[16])
        self.UtmCampaign = accountDetailsList[17]


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


# Expected format of passed string:  yyyy-mm-dd
def toDate(s):
    d = s.strip()

    # Return None if no value is present.
    if len(d) == 0:
        return None
    else:
        # print('This is the string passed to toDate: ' + s)
        year = int(d[0:4])
        month = int(d[5:7])
        day = int(d[8:10])
        r = datetime.date(year, month, day)
        return r


def toInt(s):
    i = 0
    try:
        i = int(s)
    except Exception:
        i = 0

    return i


def loadAccountFile(file):

    # Read through each line of the file.
    count = 0
    lineCount = 0
    accounts = dict()
    date = '20160607'
    fhand = open(file)
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


def printBrief(matchesDict):

    header = 'Id\t\t\t\tAcc. Name\t'
    header = header + 'User Count\tRepo Count\tCreated On\t'
    header = header + 'Trial Ends On\t'
    header = header + 'Sub. Start\tSub. End'
    print(header)

    for id in matchesDict:
        a = accounts[id]
        display = a.AccountId + '\t' + a.AccountName[0:12] + '\t\t'
        display = display + str(a.UsersCount) + '\t\t'
        display = display + str(a.ReposCount) + '\t' + a.CreatedOn + '\t'
        print(display)


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


def searchByName(rexp):
    matches = dict()
    for id in accounts:
        a = accounts[id]
        if re.search(rexp, a.AccountName.lower()):
            matches[id] = a
    return matches


def searchById(rexp):
    matches = dict()
    for id in accounts:
        if re.search(rexp, id.lower()):
            matches[id] = accounts[id]
    return matches


def getCustomerCount():
    count = 0
    for id in accounts:
        a = accounts[id]
        if a.SubscriptionPeriodStart is not None:
            if a.SubscriptionPeriodEnd >= datetime.date.today():
                count = count + 1
    return count

def getTrialAccounts():
    matches = dict()
    for id in accounts:
        a = accounts[id]
        if a.TrialEndsOn >= datetime.date.today():
            matches[id] = accounts[id]
    return matches

# Start of main processing.

# Load the file containing all the accounts.
file = '/Users/keithpijanowski/Documents/cc-data/account_export-2016-06-17T080246Z.csv'
accounts, count = loadAccountFile(file)
print('Number of accounts:  ' + str(count))

# User request loop
while True:
    command = raw_input('--> ')

    # Quit the input loop.
    if command == '-q':
        break

    if command[0:2] == '-c':
        c = getCustomerCount()
        print('Total number of customers: ' + str(c))

    # Search by name.  The search text must be a regular expression.
    if command[0:2] == '-n':
        regx = command[3:]
        matches = searchByName(regx)
        printBrief(matches)

    if command[0:6] == '-trial':
        matches = getTrialAccounts()
        printBrief(matches)

    if command[0:3] == '-id':
        regx = command[4:]
        print(regx)
        matches = searchById(regx)
        printFull(matches)
