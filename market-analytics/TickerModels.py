# Ticker Models


class Ticker:

    def __init__(self, tickerList):

        tickerIndex = 0
        dateIndex = 1
        openIndex = 2
        highIndex = 3
        lowIndex = 4
        closeIndex = 5
        volumeIndex = 6

        symbol = tickerList[tickerIndex]
        date = tickerList[dateIndex]
        openPrice = tickerList[openIndex]
        high = tickerList[highIndex]
        low = tickerList[lowIndex]
        close = tickerList[closeIndex]
        volume = tickerList[volumeIndex]

        self.Symbol = symbol
        self.Date = date
        self.Open = float(openPrice)
        self.High = float(high)
        self.Low = float(low)
        self.Close = float(close)
        self.Change = self.Close - self.Open
        self.Volume = int(volume)
