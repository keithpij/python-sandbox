#!
# User Utilities


class User:
    def __init__(self, userDetailsList):
        self.UserId = userDetailsList[0]
        self.Email = userDetailsList[1]
        self.RecieveEmails = userDetailsList[2]
        self.FullName = userDetailsList[3]
        self.CreatedOn = userDetailsList[4]
        self.LastSeenOn = userDetailsList[5]
        self.ActiveAccountMember = userDetailsList[6]
        self.ActiveAccountsCount = userDetailsList[7]
        self.TotalAccountsCount = userDetailsList[8]
        self.ActiveAccountsNotUpgradedCount = userDetailsList[9]
        self.LatestActiveAdminAccountName = userDetailsList[10]
        self.LatestActiveAdminAccountId = userDetailsList[11]
        self.LatestActiveAdminNotUpgradedAccountId = userDetailsList[12]
        self.UsageDaysCount = self.getInt(userDetailsList[13])
        self.HadAccount = userDetailsList[14]

        try:
            self.ActiveReposCount = int(userDetailsList[15])
        except Exception:
            self.ActiveReposCount = 0

        self.ActiveClassicReposCount = userDetailsList[16]
        self.ActivePlatformReposCount = userDetailsList[17]


    def getInt(self, s):
        i = 0
        try:
            i = int(s)
        except Exception:
            i = 0

        return i


# Read through each line of the file.
count = 0
lineCount = 0
users = dict()
date = '20160602'
file = '/Users/keithpijanowski/Documents/cc-data/user_export-2016-06-07T085438Z.csv'
fhand = open(file)
for line in fhand:
    lineCount = lineCount + 1

    # The first line contains the column headers.
    if lineCount == 1:
        continue

    line = line.strip()
    detailsList = line.split(',')
    u = User(detailsList)
    users[u.UserId, date] = u
    count = count + 1

    if u.ActiveReposCount > 10:
        print(u.FullName + ':  ' + str(u.ActiveReposCount))

print('Number of users:  ' + str(count))
