import numpy as np

bins=np.arange(16)
bins=np.append(bins, np.arange(17,37,2))
bins=np.append(bins, np.arange(40,75,5))
bins=np.append(bins, np.arange(80,210,10))
t=bins[:-1]
dt=bins[1:]-bins[:-1]

x=np.arange(200)
for i in range(len(t)):
    print(t[i], x[t[i]:t[i]+dt[i]])
