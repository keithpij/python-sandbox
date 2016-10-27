import sys
import datetime
import pyspark


count = 0
for line in sys.stdin:
    count += 1

print(str(datetime.date.today()) + ' Lines: ' + str(count))
