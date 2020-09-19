import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-1111, 1111, 1)
y = 2 * x**2 + 4 * x + 3
yn = 4 * x + 4
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
plt.plot(x, y, 'r-')
plt.plot(x, yn, 'b-')
plt.grid()
plt.show()


