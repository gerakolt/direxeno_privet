import numpy as np
import matplotlib.pyplot as plt

x=np.arange(10)
y=x**2
d=5

plt.plot(x,y,'k.')
plt.errorbar(x,y,d, fmt='k.')
plt.show()
