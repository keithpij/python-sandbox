# Data scrubbing Tools
import datetime


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
