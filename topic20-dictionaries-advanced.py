# Advanced Dictionary Techniques

# A dictionary of dictionaries.
# The raw data.
info = 'AAPL,20160415,112.11,112.3,109.73,109.85,46938900'

# Parse out the ticker symbol.
index = info.find(',')
tickerSymbol = info[0:index]

# Strip out the ticker symbol.
# Adding 1 to the index strips out the ',' as well.
info = info[index+1:]
index = info.find(',')

date = info[0:index]
info = info[index+1:]

tickerData = dict()
tickerData[date] = info
tickersDict[tickerSymbol] = tickerData
print(tickersDict)
print(len(tickersDict))
