# Writing files

# Open a file for output.
fout = open('portfolio.txt', 'w')

# Create a list of tickers.
tickers = 'AAPL\nAMZN\nGOOG\nMSFT\nORCL'

# Write the tickers to the file that has been opened for output.
fout.write(tickers)

# Close the file.
fout.close()
