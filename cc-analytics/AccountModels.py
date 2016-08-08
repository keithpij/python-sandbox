# Account class
import gzip
import DataTools
import csv


class Account:

    FIELD_COUNT = 0

    ACCOUNTID_HEADER = 'account_id'
    ACCOUNTNAME_HEADER = 'account_name'
    FIRSTOWNERNAME_HEADER = 'first_owner_name'
    FIRSTOWNEREMAIL_HEADER = 'first_owner_email'
    ORGANIZATIONSIZE_HEADER = 'organization_size'
    PHONENUMBER_HEADER = 'phone_number'
    PROMOCODE_HEADER = 'promo_code'
    USERSCOUNT_HEADER = 'users_count'
    SEATSCOUNT_HEADER = 'seats_count'
    AUTHORSCOUNT_HEADER = 'authors_count'
    USERQUOTA_HEADER = 'user_quota'
    REPOSCOUNT_HEADER = 'repos_count'
    PLATFORMREPOSCOUNT_HEADER = 'platform_repos_count'
    ACTIVEREPOSCOUNT_HEADER = 'active_repos_count'
    ACTIVEPLATFORMREPOSCOUNT_HEADER = 'active_platform_repos_count'
    CREATEDON_HEADER = 'created_on'
    TRIALENDSON_HEADER = 'trial_ends_on'
    PLANNAME_HEADER = 'plan_name'
    PLANPRICE_HEADER = 'plan_price'
    PLANCODE_HEADER = 'plan_code'
    BILLINGINTERVAL_HEADER = 'billing_interval'
    SUBSCRIPTIONPERIODSTART_HEADER = 'subscription_period_start'
    SUBSCRIPTIONPERIODEND_HEADER = 'subscription_period_end'
    UTMCAMPAIGN_HEADER = 'utm_campaign'

    IndexDict = dict()

    def __init__(self, accountDetailsList):

        d = Account.IndexDict
        self.AccountId = accountDetailsList[d[Account.ACCOUNTID_HEADER]]
        self.AccountName = accountDetailsList[d[Account.ACCOUNTNAME_HEADER]].strip()
        self.FirstOwnerName = accountDetailsList[d[Account.FIRSTOWNERNAME_HEADER]].strip()
        self.FirstOwnerEmail = accountDetailsList[d[Account.FIRSTOWNEREMAIL_HEADER]].strip()
        self.OrganizationSize = accountDetailsList[d[Account.ORGANIZATIONSIZE_HEADER]].strip()
        self.PhoneNumber = accountDetailsList[d[Account.PHONENUMBER_HEADER]].strip()
        self.PromoCode = accountDetailsList[d[Account.PROMOCODE_HEADER]].strip()
        self.UsersCount = DataTools.toInt(accountDetailsList[d[Account.USERSCOUNT_HEADER]])  # this could change over time
        self.ReposCount = DataTools.toInt(accountDetailsList[d[Account.REPOSCOUNT_HEADER]])  # this could change over time
        if Account.SEATSCOUNT_HEADER in d.keys():
            self.SeatCount = DataTools.toInt(accountDetailsList[d[Account.SEATSCOUNT_HEADER]])
            self.AuthorsCount = DataTools.toInt(accountDetailsList[d[Account.AUTHORSCOUNT_HEADER]])
            self.UserQuota = DataTools.toInt(accountDetailsList[d[Account.USERQUOTA_HEADER]])
            self.PlatformReposCount = DataTools.toInt(accountDetailsList[d[Account.PLATFORMREPOSCOUNT_HEADER]])
            self.ActiveReposCount = DataTools.toInt(accountDetailsList[d[Account.ACTIVEREPOSCOUNT_HEADER]])
            self.ActivePlatformReposCount = DataTools.toInt(accountDetailsList[d[Account.ACTIVEPLATFORMREPOSCOUNT_HEADER]])
        self.CreatedOn = DataTools.toDate(accountDetailsList[d[Account.CREATEDON_HEADER]])
        self.TrialEndsOn = DataTools.toDate(accountDetailsList[d[Account.TRIALENDSON_HEADER]])
        self.PlanName = accountDetailsList[d[Account.PLANNAME_HEADER]]
        self.PlanPrice = accountDetailsList[d[Account.PLANPRICE_HEADER]]
        self.PlanCode = accountDetailsList[d[Account.PLANCODE_HEADER]]
        self.BillingInterval = accountDetailsList[d[Account.BILLINGINTERVAL_HEADER]]
        self.SubscriptionPeriodStart = DataTools.toDate(accountDetailsList[d[Account.SUBSCRIPTIONPERIODSTART_HEADER]])
        self.SubscriptionPeriodEnd = DataTools.toDate(accountDetailsList[d[Account.SUBSCRIPTIONPERIODEND_HEADER]])
        self.UtmCampaign = accountDetailsList[d[Account.UTMCAMPAIGN_HEADER]]
        # Need fields for:
        # credit card churn date
        # customer cancellation date
        # cancellation reason
        # SCM
        # Platform Repo count
        # Classic Repo count


def setIndecies(headerList):
    # The field count from the header line will help to correct scenarios
    # where the user put a comma in the Account Name field.
    global FIELD_COUNT
    FIELD_COUNT = len(headerList)

    # By not hard coding these value we are insulated from changes to the
    # underlying export file.
    for i in range(0, FIELD_COUNT):
        Account.IndexDict[headerList[i]] = i


def loadAccountFile(file):

    # Setup needed variables
    accountCount = 0
    lineCount = 0
    accounts = dict()
    fhand = gzip.open(file)
    reader = csv.reader(fhand)

    # Read through each line of the file.
    for line in reader:
        lineCount = lineCount + 1

        # The first line contains the column headers.
        if lineCount == 1:
            setIndecies(line)
        else:
            a = Account(line)
            accounts[a.AccountId] = a
            accountCount = accountCount + 1

    return accounts, accountCount
