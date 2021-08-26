'''
Plotting a simple face using complex numbers.
'''

import matplotlib.pyplot as plt 

face = {2+2j, 3+2j, 1.75+1j, 2+1j, 2.25+1j, 2.5+1j, 2.75+1j, 3+1j, 3.25+1j}

fig, ax = plt.subplots()
for c in face:
    ax.scatter(c.real, c.imag, s=5, cmap=None)

ax.grid(True, which='both')
ax.set_aspect('equal')

# set the x-spine (see below for more info on `set_position`)
#ax.spines['left'].set_position('zero')

# turn off the right spine/ticks
#ax.spines['right'].set_color('none')
#ax.yaxis.tick_left()

# set the y-spine
#ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
#ax.spines['top'].set_color('none')
#ax.xaxis.tick_bottom()

limit = 10
plt.xlim((-limit, limit))
plt.ylim((-limit, limit))

plt.ylabel('Imaginary')
plt.xlabel('Real')

plt.show()
#plot(face)
