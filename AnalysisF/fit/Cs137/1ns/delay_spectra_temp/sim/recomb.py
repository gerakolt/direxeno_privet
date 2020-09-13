import matplotlib.pyplot as plt
import numpy as np

eta=0.7
a=0.002
ni=np.ones(200*100)
ne=np.ones(200*100)*(1-eta)
for i in range(1, 200*100):
    ni[i]=ni[i-1]*(1-a*ne[i-1])
    ne[i]=ne[i-1]*(1-a*ni[i-1])

p=np.sum(a*(ni*ne).reshape(200,100), axis=1)/(1-eta)

print(np.sum(p))
plt.figure()
plt.plot(p, '.k')
# plt.plot(ni, '.g')
# plt.plot(ne, '.r')

plt.show()
