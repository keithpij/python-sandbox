# Data scrubbing Tools
import datetime


# Expected format of passed string:  yyyy-mm-dd
def toDateDash(s):
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


# Expected format of passed string:  yyyy-mm-dd
def to_date_slash(string_date):
    string_date = string_date.strip()

    # Return None if no value is present.
    if len(string_date) == 0:
        return None
    else:
        # print('This is the string passed to toDate: ' + s)
        pieces = string_date.split('/')
        month = int(pieces[0])
        day = int(pieces[1])
        year = int(pieces[2])
        return_date = datetime.date(year, month, day)
        return return_date


# Expected format of passed string:  yyyymmdd
def toDate(s):
    d = s.strip()

    # Return None if no value is present.
    if len(d) == 0:
        return None
    else:
        # print('This is the string passed to toDate: ' + s)
        year = int(d[0:4])
        month = int(d[4:6])
        day = int(d[6:8])
        r = datetime.date(year, month, day)
        return r


def toInt(s):
    i = 0
    try:
        i = int(s)
    except Exception:
        i = 0
    return i


def toFloat(s):
    f = 0
    if len(s) < 1:
        return 0
    if s[0] == '$':
        s = s[1:]
    try:
        f = float(s)
    except Exception:
        f = 0
    return f
