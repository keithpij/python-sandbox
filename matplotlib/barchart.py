'''
A simple bar chart with Matplotlib.
'''

import numpy as np
from matplotlib import pyplot as plt

x = ('3/15/1999', '12/4/2012', '5/9/2007', '8/16/1998', '7/25/2001', '12/31/2011', '6/25/1993', '2/16/1995', '2/10/1981', '5/3/2014' )
y = (1, 12, 33, 44, 55, 76, 87, 118, 119, 145)

fig = plt.figure()

width = 5
ind = np.arange(len(y))
print(ind)
plt.bar(ind, y) # the bar function has a width parameter that I could not get to work properly.
plt.xticks(ind, x)

fig.autofmt_xdate()
#plt.savefig('figure.pdf')
plt.show()
