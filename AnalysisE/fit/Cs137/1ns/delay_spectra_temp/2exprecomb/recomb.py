import matplotlib.pyplot as plt
import numpy as np

eta=0.5
a=0.005

ni=np.ones(200*100)
ne=np.ones(200*100)*eta

for i in range(1, 200*100):
    ni[i]=ni[i-1]*(1-a*ne[i-1])
    ne[i]=ne[i-1]*(1-a*ni[i-1])

plt.figure()
plt.plot(-a*(ni[1:]-ni[:-1]), '-.k')
plt.show()
